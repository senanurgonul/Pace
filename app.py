from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import holidays
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
son_tahminler = []  # Excel çıktısı için global değişken

def load_and_prepare_data():
    df = pd.read_csv("denemedata2.csv", sep=';', encoding='utf-8')
    df['Tarih'] = pd.to_datetime(df['Tarih'].astype(str), format="%d%m%Y", errors='coerce')
    df = df.dropna(subset=['Tarih'])

    df['Gün'] = df['Tarih'].dt.weekday
    df['Ay'] = df['Tarih'].dt.month
    df['Hafta'] = df['Tarih'].dt.isocalendar().week
    tr_holidays = holidays.Turkey(years=[2022, 2023, 2024])
    df['Tatil'] = df['Tarih'].apply(lambda x: x.weekday() >= 5 or x in tr_holidays).astype(int)

    return df

def egit_modeller(df):
    features = ['Gün', 'Ay', 'Hafta', 'Tatil']
    X = df[features]

    model_davet = RandomForestRegressor(n_estimators=100, random_state=42)
    model_teyit = RandomForestRegressor(n_estimators=100, random_state=42)
    model_teyit_yok = RandomForestRegressor(n_estimators=100, random_state=42)
    model_katilim = RandomForestRegressor(n_estimators=100, random_state=42)

    model_davet.fit(X, df['Davet Edilen Aday Sayısı'])
    model_teyit.fit(X, df['Teyit Veren Aday Sayısı'])
    model_teyit_yok.fit(X, df['Teyit Vermeyen Aday Sayısı'])
    model_katilim.fit(X, df['Sınava Katılan Aday Sayısı'])

    return model_davet, model_teyit, model_teyit_yok, model_katilim

def resmi_tatil_mi(tarih):
    tr_holidays = holidays.Turkey(years=tarih.year)
    return tarih.weekday() >= 5 or tarih in tr_holidays

def tahmin_et(model_davet, model_teyit, model_teyit_yok, model_katilim, baslangic, bitis, tolerans_aktif=False):
    baslangic = datetime.strptime(baslangic, "%Y-%m-%d")
    bitis = datetime.strptime(bitis, "%Y-%m-%d")
    tahmin_sonuclari = []

    hedef_katilim = 45 * 0.9  # %90 kapasite hedefi (40.5 kişi)
    maksimum_katilim = 45     # Asla 45'i geçmesin

    mevcut_tarih = baslangic
    while mevcut_tarih <= bitis:
        if not resmi_tatil_mi(mevcut_tarih):
            gun = mevcut_tarih.weekday()
            ay = mevcut_tarih.month
            hafta = mevcut_tarih.isocalendar().week
            tatil = 0

            X = pd.DataFrame([[gun, ay, hafta, tatil]], columns=['Gün', 'Ay', 'Hafta', 'Tatil'])
            orijinal_davet = model_davet.predict(X)[0]

            en_iyi_davet = 20
            en_yakin_katilim = 0
            en_fark = float('inf')

            for aday_davet in range(20, 150):
                katsayi = aday_davet / orijinal_davet if orijinal_davet else 1
                tahmini_katilim = model_katilim.predict(X)[0] * katsayi

                if tahmini_katilim <= maksimum_katilim:
                    fark = abs(tahmini_katilim - hedef_katilim)
                    if fark < en_fark:
                        en_yakin_katilim = tahmini_katilim
                        en_iyi_davet = aday_davet
                        en_fark = fark

            # Seçilen en iyi davet sayısına göre tahminleri hesapla
            katsayi = en_iyi_davet / orijinal_davet if orijinal_davet else 1
            teyit = int(round(model_teyit.predict(X)[0] * katsayi))
            teyit_vermeyen = int(round(model_teyit_yok.predict(X)[0] * katsayi))
            katilim = int(round(model_katilim.predict(X)[0] * katsayi))

            tahmin_sonuclari.append({
                "Tarih": mevcut_tarih.strftime("%Y-%m-%d"),
                "Gün": mevcut_tarih.strftime("%A"),
                "Davet Edilen": en_iyi_davet,
                "Teyit Veren": teyit,
                "Teyit Vermeyen": teyit_vermeyen,
                "Sınava Katılacak": katilim
            })

        mevcut_tarih += timedelta(days=1)

    return tahmin_sonuclari



def grafik_uret(tahminler):
    df = pd.DataFrame(tahminler)

    if df.empty:
        return None, None, None

    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values("Tarih")
    df["Kümülatif Katılım"] = df["Sınava Katılacak"].cumsum()
    df["Salon Kullanımı"] = df["Sınava Katılacak"] / 45

    def grafik_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    fig1, ax1 = plt.subplots()
    ax1.plot(df["Tarih"], df["Davet Edilen"], label="Davet", marker='o')
    ax1.plot(df["Tarih"], df["Teyit Veren"], label="Teyit", marker='o')
    ax1.plot(df["Tarih"], df["Sınava Katılacak"], label="Katılım", marker='o')
    ax1.set_title("Günlük Aday Dağılımı")
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(df["Tarih"], df["Kümülatif Katılım"], color='green', marker='o')
    ax2.set_title("Kümülatif Beklenen Katılım")
    fig2.tight_layout()

    fig3, ax3 = plt.subplots()
    ax3.bar(df["Tarih"].dt.strftime("%d-%m"), df["Salon Kullanımı"])
    ax3.axhline(1.0, color='red', linestyle='--', label='Tam kapasite')
    ax3.set_title("Salon Kullanım Oranı")
    ax3.set_ylim(0, 1.5)
    ax3.legend()
    fig3.tight_layout()

    return grafik_base64(fig1), grafik_base64(fig2), grafik_base64(fig3)

@app.route("/", methods=["GET", "POST"])
def index():
    global son_tahminler
    tahminler = []
    secilen_aralik = None
    grafik1 = grafik2 = grafik3 = None

    if request.method == "POST":
        baslangic = request.form["start_date"]
        bitis = request.form["end_date"]
        tolerans_aktif = request.form.get("tolerans") == "yes"
        secilen_aralik = f"{baslangic} → {bitis}"

        df = load_and_prepare_data()
        model_davet, model_teyit, model_teyit_yok, model_katilim = egit_modeller(df)
        tahminler = tahmin_et(model_davet, model_teyit, model_teyit_yok, model_katilim, baslangic, bitis, tolerans_aktif)
        grafik1, grafik2, grafik3 = grafik_uret(tahminler)
        son_tahminler = tahminler

    return render_template("index.html", tahminler=tahminler, secilen_aralik=secilen_aralik,
                           grafik1=grafik1, grafik2=grafik2, grafik3=grafik3)

@app.route("/indir")
def indir():
    if not son_tahminler:
        return "Henüz tahmin yapılmadı.", 400

    df = pd.DataFrame(son_tahminler)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Tahminler")
    output.seek(0)
    return send_file(output, download_name="tahminler.xlsx", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)