# app.py
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# --- Fungsi Pemodelan dan Simulasi Sistem Dinamik ---
def run_simulation(df, policy_prod_change, policy_cons_change, projection_years=5):
    
    # 1. PRE-PROCESSING DATA
    df.columns = df.columns.str.strip() # Bersihkan spasi di nama kolom (Mencegah KeyError)
    df = df.apply(pd.to_numeric, errors='coerce').dropna() # Konversi ke numerik

    if len(df) < 2:
        return {"error": "Data terlalu sedikit untuk simulasi (min 2 baris historis)."}, None, None

    # Tentukan tahun awal dan tahun terakhir historis
    last_year = df['year'].max()
    years_to_project = [last_year + i for i in range(1, projection_years + 1)]
    
    # 2. MENGHITUNG RATA-RATA TINGKAT PERTUMBUHAN HISTORIS (Growth Rate)
    # Gunakan data awal dan akhir untuk menghitung CAGR (Compound Annual Growth Rate)
    
    n_periods = df['year'].nunique() - 1
    
    def calculate_cagr(series):
        if series.iloc[0] == 0: return 0.05 # Hindari pembagian nol
        cagr = (series.iloc[-1] / series.iloc[0])**(1/n_periods) - 1
        return cagr

    # Rata-rata Pertumbuhan Historis
    rate_prod = calculate_cagr(df['production_ton'])
    rate_pop = calculate_cagr(df['population_thousand'])
    rate_cons_pc = calculate_cagr(df['consumption_per_capita_kg'])
    
    # Ambil nilai awal untuk proyeksi
    initial_prod = df['production_ton'].iloc[-1]
    initial_pop = df['population_thousand'].iloc[-1]
    initial_cons_pc = df['consumption_per_capita_kg'].iloc[-1]
    initial_price = df['price_rp_per_kg'].iloc[-1]

    # Terapkan Kebijakan (Policy) ke tingkat pertumbuhan
    rate_prod_sim = rate_prod * (1 + policy_prod_change)
    rate_cons_pc_sim = rate_cons_pc * (1 + policy_cons_change)
    
    # 3. PROYEKSI DAN SIMULASI (5 TAHUN KE DEPAN)
    projection_data = []
    current_prod = initial_prod
    current_pop = initial_pop
    current_cons_pc = initial_cons_pc
    
    for year in years_to_project:
        # Proyeksi (Growth Model: Value_t = Value_t-1 * (1 + rate))
        current_prod *= (1 + rate_prod_sim)
        current_pop *= (1 + rate_pop) # Populasi diasumsikan tidak terpengaruh kebijakan
        current_cons_pc *= (1 + rate_cons_pc_sim)
        
        # Total Konsumsi Nasional (Demand)
        total_demand = current_cons_pc * current_pop / 1000 # Bagi 1000 karena pop dalam ribuan
        
        # Perhitungan Surplus/Defisit
        surplus_deficit = current_prod - total_demand
        
        # MODEL HARGA CAUSAL (System Dynamics Logic)
        # Harga bergerak melawan Surplus/Defisit.
        # Defisit (negatif) -> Harga naik. Surplus (positif) -> Harga turun.
        # Persentase Surplus/Defisit (terhadap Produksi)
        s_d_perc = surplus_deficit / current_prod
        
        # Harga Baru: Harga_t = Harga_t-1 * (1 - FaktorKoreksi)
        # Faktor Koreksi: Harga disesuaikan 0.5% untuk setiap 1% S/D
        # (Defisit 5% -> Harga naik 2.5%; Surplus 5% -> Harga turun 2.5%)
        # Rumus: Harga_t = initial_price * (1 - (s_d_perc * 0.5))
        # Agar harga bergerak dari waktu ke waktu (dinamik), gunakan harga tahun sebelumnya:
        if projection_data:
            current_price = projection_data[-1]['price'] * (1 - (s_d_perc * 0.5))
        else:
            current_price = initial_price * (1 - (s_d_perc * 0.5))

        
        projection_data.append({
            'year': year,
            'supply': current_prod,
            'demand': total_demand,
            'surplus_deficit': surplus_deficit,
            'price': current_price
        })

    # Gabungkan data historis dan proyeksi untuk visualisasi
    history = df[['year', 'production_ton', 'consumption_per_capita_kg', 'population_thousand', 'price_rp_per_kg']].copy()
    history['demand'] = history['consumption_per_capita_kg'] * history['population_thousand'] / 1000
    history['surplus_deficit'] = history['production_ton'] - history['demand']
    
    proj_df = pd.DataFrame(projection_data)
    
    # Ubah nama kolom agar konsisten
    history = history.rename(columns={'production_ton': 'supply', 'price_rp_per_kg': 'price'})
    history['type'] = 'Historis'
    proj_df['type'] = 'Proyeksi'
    
    full_df = pd.concat([history[['year', 'supply', 'demand', 'surplus_deficit', 'price', 'type']], proj_df])
    
    # 4. PEMBUATAN GRAFIK
    plots = {}
    
    # --- Grafik 1: Supply & Demand (Proyeksi Ketersediaan) ---
    plt.figure(figsize=(10, 6))
    plt.plot(full_df['year'], full_df['supply'], label='Supply (Produksi)', marker='o', linestyle='-', color='green')
    plt.plot(full_df['year'], full_df['demand'], label='Demand (Kebutuhan Total)', marker='x', linestyle='--', color='red')
    
    plt.axvline(x=last_year, color='gray', linestyle=':', linewidth=1)
    plt.text(last_year + 0.1, full_df['supply'].max(), 'Titik Simulasi', color='gray')
    
    plt.title('Proyeksi Supply (Produksi) dan Demand (Kebutuhan) Daging Ayam')
    plt.xlabel('Tahun')
    plt.ylabel('Kuantitas (Ribu Ton)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plots['supply_demand'] = save_plot_to_base64(plt)
    
    # --- Grafik 2: Surplus/Defisit dan Harga ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Tahun')
    ax1.set_ylabel('Surplus/Defisit (Ribu Ton)', color=color)
    ax1.bar(full_df['year'], full_df['surplus_deficit'], color=np.where(full_df['surplus_deficit'] >= 0, 'lightgreen', 'salmon'), label='Surplus/Defisit')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

    ax2 = ax1.twinx()  # instan kedua berbagi sumbu x yang sama
    color = 'tab:red'
    ax2.set_ylabel('Harga (Rp/kg)', color=color)
    ax2.plot(full_df['year'], full_df['price'], color=color, marker='D', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Hubungan Surplus/Defisit dengan Stabilitas Harga')
    plots['sd_price'] = save_plot_to_base64(plt)


    # 5. RINGKASAN HASIL
    results = {
        "scenario_name": f"Produksi Policy: {policy_prod_change*100:+.0f}% | Konsumsi Policy: {policy_cons_change*100:+.0f}%",
        "last_historical_year": int(last_year),
        "proj_start_price": f"Rp {initial_price:,.0f}",
        "proj_end_price": f"Rp {proj_df['price'].iloc[-1]:,.0f}",
        "surplus_2025": f"{proj_df['surplus_deficit'].iloc[0]:,.2f} Ribu Ton",
        "surplus_end": f"{proj_df['surplus_deficit'].iloc[-1]:,.2f} Ribu Ton"
    }
    
    return results, plots, full_df.to_dict('records')

def save_plot_to_base64(plt):
    """Fungsi pembantu untuk menyimpan plot ke string base64."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"

# --- Routing Aplikasi Web ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ambil input kebijakan (policy simulation) dari form
        policy_prod = float(request.form.get('policy_prod_change', 0)) / 100
        policy_cons = float(request.form.get('policy_cons_change', 0)) / 100
        
        # Logika pemuatan data (File Upload atau Manual Input)
        try:
            if 'file' in request.files and request.files['file'].filename != '':
                df = pd.read_csv(request.files['file'])
            elif request.form.get('manual_data'):
                manual_data = request.form['manual_data'].strip()
                data_list = [line.split(',') for line in manual_data.split('\n') if line.strip()]
                columns = ['year', 'production_ton', 'consumption_per_capita_kg', 'population_thousand', 'price_rp_per_kg']
                df = pd.DataFrame(data_list, columns=columns)
            else:
                return render_template('index.html', error="Mohon unggah file atau masukkan data manual.")

            # Jalankan simulasi
            results, plots, full_data = run_simulation(df, policy_prod, policy_cons)

            if "error" in results:
                return render_template('index.html', error=results["error"])

            return render_template('result.html', 
                                   results=results, 
                                   plots=plots, 
                                   full_data=full_data)
            
        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan saat pemrosesan data atau simulasi. Detail: {e}")

    return render_template('index.html', error=None)

if __name__ == '__main__':
    # Untuk UTS, disarankan untuk mengatur debug=False sebelum deployment akhir
    app.run(debug=True)