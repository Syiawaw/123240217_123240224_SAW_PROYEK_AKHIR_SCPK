import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="SPK Wisata Indonesia – SAW",
    page_icon="",
    layout="wide",
)

# WARNA TEMA
PALETTE = ["#00B3FF", "#FBFF00", "#FF9900", "#FF3300", "#FF006F", "#00FF3C"]
sns.set_theme(style="whitegrid", palette=PALETTE)

# FUNGSI BANTU
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Muat dataset dan lakukan preprocessing awal."""
    df = pd.read_csv(path)

    # Buang kolom kosong / tidak relevan
    df = df.drop(columns=["Unnamed: 11", "Unnamed: 12"], errors="ignore")

    # Isi Time_Minutes yang kosong dengan median (60 menit)
    df["Time_Minutes"] = df["Time_Minutes"].fillna(df["Time_Minutes"].median())

    # ── Kriteria 4 : Skor Kategori (1–6, default merata)
    cat_rank = {
        "Taman Hiburan": 6,
        "Budaya": 5,
        "Cagar Alam": 4,
        "Bahari": 3,
        "Tempat Ibadah": 2,
        "Pusat Perbelanjaan": 1,
    }
    df["Category_Score"] = df["Category"].map(cat_rank).fillna(1)

    # ── Kriteria 5 : Skor Popularitas Kota (jumlah destinasi per kota)
    city_pop = df["City"].value_counts().to_dict()
    df["City_Popularity"] = df["City"].map(city_pop)

    return df


def build_category_score(df: pd.DataFrame, fav_category: str) -> pd.Series:
    """
    Hitung ulang Category_Score berdasarkan pilihan user.
    Kategori favorit user = 6, sisanya turun berurutan.
    """
    all_cats = [
        "Taman Hiburan", "Budaya", "Cagar Alam",
        "Bahari", "Tempat Ibadah", "Pusat Perbelanjaan",
    ]
    # Rotasi urutan sehingga fav_category berada di posisi pertama (nilai 6)
    reordered = [fav_category] + [c for c in all_cats if c != fav_category]
    rank_map = {cat: 6 - i for i, cat in enumerate(reordered)}
    return df["Category"].map(rank_map).fillna(1)


def normalize_saw(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """
    Normalisasi matriks keputusan menggunakan rumus SAW dengan NumPy.

    Benefit : r_ij = x_ij / max(x_j)      → np.max()
    Cost    : r_ij = min(x_j) / x_ij      → np.min(), np.where()
              Khusus jika min = 0 (misal harga gratis):
              r_ij = 1.0  jika x_ij == 0  (gratis = terbaik)
              r_ij = min_nonzero / x_ij   untuk nilai > 0
    """
    norm = pd.DataFrame(index=df.index)
    for col, ctype in criteria.items():
        arr = df[col].to_numpy(dtype=float)          # ← konversi ke NumPy array

        if ctype == "benefit":
            col_max = np.max(arr)                    # ← np.max()
            norm[col] = np.where(                    # ← np.where()
                col_max == 0,
                0.0,
                arr / col_max
            )
        else:  # cost
            col_min = np.min(arr)                    # ← np.min()
            if col_min == 0:
                # Ambil nilai minimum yang bukan nol menggunakan np.where + np.min
                nonzero_vals = arr[np.where(arr > 0)]    # ← np.where sebagai mask
                min_nonzero  = np.min(nonzero_vals) if len(nonzero_vals) > 0 else 1.0
                # np.where: jika 0 → 1.0 (gratis=sempurna), jika >0 → min/x
                with np.errstate(divide="ignore", invalid="ignore"):   # ← np.errstate()
                    norm[col] = np.where(
                        arr == 0,
                        1.0,
                        min_nonzero / arr
                    )
            else:
                norm[col] = col_min / arr            # ← operasi vektor NumPy

    return norm


def compute_saw(df: pd.DataFrame, bobot: dict, fav_cat: str) -> pd.DataFrame:
    """
    Hitung nilai SAW menggunakan operasi matriks NumPy.
    V_i = Σ w_j * r_ij  →  np.dot(matriks_norm, vektor_bobot)
    """
    data = df.copy()

    # Update Category_Score berdasarkan pilihan user
    data["Category_Score"] = build_category_score(data, fav_cat)

    criteria_cols = {
        "Rating":          "benefit",
        "Price":           "cost",
        "Time_Minutes":    "benefit",
        "Category_Score":  "benefit",
        "City_Popularity": "benefit",
    }

    # Normalisasi → hasilnya DataFrame
    norm_df = normalize_saw(data, criteria_cols)

    # Bobot dinormalisasi menggunakan NumPy
    bobot_arr  = np.array([bobot[k] for k in criteria_cols], dtype=float)  # ← np.array()
    bobot_norm = bobot_arr / np.sum(bobot_arr)                              # ← np.sum()
    norm_weights = dict(zip(criteria_cols.keys(), bobot_norm))

    # Hitung V_i = dot product matriks normalisasi × vektor bobot
    # np.dot() jauh lebih efisien daripada loop manual
    matriks_norm = norm_df[list(criteria_cols.keys())].to_numpy(dtype=float)  # ← np.array 2D
    nilai_saw_arr = np.dot(matriks_norm, bobot_norm)                           # ← np.dot()

    # Ganti NaN/inf dengan 0 menggunakan np.nan_to_num
    nilai_saw_arr = np.nan_to_num(nilai_saw_arr, nan=0.0, posinf=0.0)          # ← np.nan_to_num()

    data["Nilai_SAW"] = nilai_saw_arr

    # Peringkat menggunakan np.argsort
    # argsort descending → urutan indeks dari nilai terbesar ke terkecil
    urutan = np.argsort(-nilai_saw_arr)                                        # ← np.argsort()
    peringkat = np.empty_like(urutan)
    peringkat[urutan] = np.arange(1, len(urutan) + 1)                         # ← np.arange()
    data["Peringkat"] = peringkat

    data = data.sort_values("Peringkat")
    return data, norm_df, norm_weights

# MUAT DATA
DATA_PATH = "tourism_with_id.csv"

try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        "File `tourism_with_id.csv` tidak ditemukan. "
        "Pastikan file berada di folder yang sama dengan script ini."
    )
    st.stop()

# SIDEBAR  ─  NAVIGASI & INPUT BOBOT
with st.sidebar:
    st.image(
        "https://kab-jayawijaya.kpu.go.id/public/kab-jayawijaya/images/1760966694_dc3d3c79c4bfdc81974e.jpg",
        width=250,
    )
    st.title("SPK Wisata Indonesia")
    st.divider()

    halaman = st.radio(
        "Navigasi",
        ["Beranda", "Dataset", "Hitung SPK", "Visualisasi", "Profil Kelompok"],
        index=0,
    )
    st.divider()

    # Input Bobot (widget interaktif) 
    st.subheader("Bobot Kriteria")
    st.caption("Geser slider untuk mengatur bobot tiap kriteria.")

    w_rating   = st.slider("Rating",              min_value=1, max_value=10, value=5)
    w_price    = st.slider("Harga Tiket",          min_value=1, max_value=10, value=4)
    w_time     = st.slider("Durasi Kunjungan",     min_value=1, max_value=10, value=3)
    w_category = st.slider("Preferensi Kategori", min_value=1, max_value=10, value=4)
    w_city     = st.slider("Popularitas Kota",     min_value=1, max_value=10, value=2)

    bobot = {
        "Rating":          w_rating,
        "Price":           w_price,
        "Time_Minutes":    w_time,
        "Category_Score":  w_category,
        "City_Popularity": w_city,
    }

    total_w = sum(bobot.values())
    st.caption(f"Total bobot saat ini: **{total_w}** (akan dinormalisasi otomatis)")
    st.divider()

    # Preferensi Kategori (Selectbox) 
    st.subheader("Preferensi Kategori")
    fav_category = st.selectbox(
        "Pilih kategori wisata favorit Anda:",
        options=[
            "Taman Hiburan", "Budaya", "Cagar Alam",
            "Bahari", "Tempat Ibadah", "Pusat Perbelanjaan",
        ],
        index=0,
    )

    #Filter Kota (Multiselect)
    st.subheader("Filter Kota")
    all_cities = sorted(df_raw["City"].unique().tolist())
    selected_cities = st.multiselect(
        "Tampilkan kota:",
        options=all_cities,
        default=all_cities,
    )
    if not selected_cities:
        selected_cities = all_cities  # default semua

    st.divider()

# FILTER DATA BERDASARKAN KOTA
df_filtered = df_raw[df_raw["City"].isin(selected_cities)].copy()

# HALAMAN : BERANDA
if halaman == "Beranda":
    st.title("Sistem Pendukung Keputusan")
    st.subheader("Rekomendasi Tempat Wisata Terbaik di Indonesia")
    st.markdown(
        """
        Aplikasi ini membantu Anda menemukan tempat wisata terbaik menggunakan
        metode **Simple Additive Weighting (SAW)**.

        ### Tentang Dataset
        Dataset berisi informasi **437 destinasi wisata** di 5 kota besar Indonesia.

        ### 5 Kriteria Penilaian
        | No | Kriteria | Tipe | Keterangan |
        |:--:|----------|:----:|------------|
        | 1 |**Rating** | Benefit | Penilaian rata-rata dari pengunjung (3.4 – 5.0) |
        | 2 |**Harga Tiket** | Cost | Tiket masuk dalam Rupiah – semakin murah = lebih baik |
        | 3 |**Durasi Kunjungan** | Benefit | Estimasi waktu kunjungan (menit) |
        | 4 |**Preferensi Kategori** | Benefit | Skor berdasarkan kategori favorit Anda |
        | 5 |**Popularitas Kota** | Benefit | Jumlah destinasi di kota tersebut |

        ### Rumus SAW
        """
    )
    st.latex(r"V_i = \sum_{j=1}^{n} w_j \cdot r_{ij}")
    st.markdown(
        """
        Di mana:
        - $V_i$ = Nilai preferensi alternatif ke-$i$
        - $w_j$ = Bobot kriteria ke-$j$ (sudah dinormalisasi)
        - $r_{ij}$ = Nilai ternormalisasi alternatif $i$ pada kriteria $j$

        **Normalisasi :**
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Kriteria Benefit**")
        st.latex(r"r_{ij} = \frac{x_{ij}}{\max_i(x_{ij})}")
    with col2:
        st.markdown("**Kriteria Cost**")
        st.latex(r"r_{ij} = \frac{\min_i(x_{ij})}{x_{ij}}")

    st.divider()

    # Statistik ringkas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Destinasi", f"{len(df_raw):,}")
    col2.metric("Kota", df_raw["City"].nunique())
    col3.metric("Kategori", df_raw["Category"].nunique())
    col4.metric("Rata-rata Rating", f"{df_raw['Rating'].mean():.2f}")

# HALAMAN : DATASET
elif halaman == "Dataset":
    st.title("Dataset Tempat Wisata Indonesia")

    st.info(
        f"Menampilkan **{len(df_filtered):,} baris** data "
        f"dari kota: {', '.join(selected_cities)}"
    )

    # Tabel interaktif
    tampil_cols = [
        "Place_Id", "Place_Name", "Category", "City",
        "Price", "Rating", "Time_Minutes",
    ]
    st.dataframe(
        df_filtered[tampil_cols].rename(columns={
            "Place_Id":     "ID",
            "Place_Name":   "Nama Tempat",
            "Category":     "Kategori",
            "City":         "Kota",
            "Price":        "Harga (Rp)",
            "Rating":       "Rating",
            "Time_Minutes": "Durasi (menit)",
        }),
        use_container_width=True,
        height=450,
    )

    st.divider()
    
# HALAMAN : HITUNG SPK
elif halaman == "Hitung SPK":
    st.title("Perhitungan SPK – Metode SAW")

    # Tampilkan bobot yang dipilih
    st.subheader("Konfigurasi Bobot")
    total_w = sum(bobot.values())
    norm_w  = {k: round(v / total_w, 4) for k, v in bobot.items()}

    bobot_df = pd.DataFrame({
        "Kriteria":  ["Rating", "Harga Tiket", "Durasi Kunjungan", "Pref. Kategori", "Popularitas Kota"],
        "Tipe":      ["Benefit", "Cost", "Benefit", "Benefit", "Benefit"],
        "Bobot Asli": list(bobot.values()),
        "Bobot Norm.": list(norm_w.values()),
    })
    st.dataframe(bobot_df, use_container_width=True, hide_index=True)

    st.markdown(f"**Kategori Favorit Anda:** `{fav_category}`")
    st.divider()

    # Tombol eksekusi 
    if st.button("Jalankan Perhitungan SAW", type="primary", use_container_width=True):
        if df_filtered.empty:
            st.warning("Tidak ada data untuk dihitung. Periksa filter kota Anda.")
            st.stop()

        with st.spinner("Sedang menghitung nilai SAW..."):
            df_result, norm_df, norm_w_used = compute_saw(df_filtered, bobot, fav_category)

        st.success("Perhitungan selesai!")
        st.divider()

        # ── Step 1 : Matriks Keputusan Awal ──
        st.subheader("Langkah 1 – Matriks Keputusan (10 Teratas)")
        st.caption("Nilai asli dari tiap alternatif untuk setiap kriteria.")
        raw_cols = ["Place_Name", "Rating", "Price", "Time_Minutes", "Category_Score", "City_Popularity"]

        matriks_df = df_result[raw_cols].head(10).copy()

        # Format kolom Harga: tampilkan "Gratis" jika 0, "Rp X,XXX" jika > 0
        matriks_df["Price"] = matriks_df["Price"].apply(
            lambda x: "Gratis" if x == 0 else f"Rp {int(x):,}"
        )
        # Format durasi: tampilkan sebagai "X menit"
        matriks_df["Time_Minutes"] = matriks_df["Time_Minutes"].apply(
            lambda x: f"{int(x)} menit"
        )

        st.dataframe(
            matriks_df.rename(columns={
                "Place_Name":      "Nama Tempat",
                "Rating":          "Rating",
                "Price":           "Harga Tiket",
                "Time_Minutes":    "Durasi",
                "Category_Score":  "Skor Kategori",
                "City_Popularity": "Popularitas Kota",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("**Catatan:** *Gratis* = harga tiket Rp 0 (masuk gratis). "
                   "Pada normalisasi SAW, harga Rp 0 mendapat nilai **1.0** (terbaik) "
                   "karena kriteria Harga bersifat Cost — semakin murah semakin baik.")

        # Step 2 : Matriks Ternormalisasi
        st.subheader("Langkah 2 – Matriks Normalisasi (10 Teratas)")
        st.caption("Nilai setiap kriteria setelah dinormalisasi dengan rumus SAW.")
        norm_show = norm_df.loc[df_result.index].head(10).copy()
        norm_show.insert(0, "Nama Tempat", df_result["Place_Name"].values[:10])
        norm_show = norm_show.rename(columns={
            "Rating":          "Rating (norm)",
            "Price":           "Harga (norm)",
            "Time_Minutes":    "Durasi (norm)",
            "Category_Score":  "Kategori (norm)",
            "City_Popularity": "Popularitas (norm)",
        })
        st.dataframe(norm_show.round(4), use_container_width=True, hide_index=True)

        # Step 3 : Nilai SAW & Bobot 
        st.subheader("Langkah 3 – Bobot Ternormalisasi yang Digunakan")
        w_df = pd.DataFrame({
            "Kriteria": ["Rating", "Harga Tiket", "Durasi", "Pref. Kategori", "Popularitas Kota"],
            "Bobot (w)": [round(norm_w_used[k], 4) for k in
                          ["Rating", "Price", "Time_Minutes", "Category_Score", "City_Popularity"]],
        })
        st.dataframe(w_df, use_container_width=True, hide_index=True)

        # ── Step 4 : Tabel Hasil Perangkingan
        st.subheader("Langkah 4 – Hasil Perangkingan Akhir")
        st.caption("Diurutkan dari nilai tertinggi (Peringkat 1) hingga terendah.")

        result_cols = [
            "Peringkat", "Place_Name", "Category", "City",
            "Price", "Rating", "Time_Minutes", "Nilai_SAW",
        ]
        result_display = df_result[result_cols].rename(columns={
            "Peringkat":    "Peringkat",
            "Place_Name":   "Nama Tempat",
            "Category":     "Kategori",
            "City":         "Kota",
            "Price":        "Harga (Rp)",
            "Rating":       "Rating",
            "Time_Minutes": "Durasi (menit)",
            "Nilai_SAW":    "Nilai SAW",
        })
        result_display["Nilai SAW"] = result_display["Nilai SAW"].round(4)

        st.dataframe(result_display, use_container_width=True, hide_index=True, height=500)

        # Simpan ke session state untuk halaman visualisasi
        st.session_state["df_result"] = df_result

        # Top 5 sorotan 
        st.divider()
        st.subheader("Top 5 Rekomendasi Terbaik")
        top5 = df_result.head(5)
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                st.markdown(f"**{row['Place_Name']}**")
                st.markdown(f"{row['City']}")
                st.markdown(f"{row['Category']}")
                st.markdown(f"{row['Rating']}")
                st.markdown(f"Rp {int(row['Price']):,}")
                st.markdown(f"SAW: `{row['Nilai_SAW']:.4f}`")

    else:
        st.info("Atur bobot di sidebar, lalu tekan tombol **Jalankan Perhitungan SAW** untuk melihat hasil.")

# HALAMAN : VISUALISASI
elif halaman == "Visualisasi":
    st.title("Visualisasi Data Analitik")
    # Grafik 1 : plt.bar() – Jumlah Destinasi per Kota 
    st.subheader("Grafik 1 – Jumlah Destinasi Wisata per Kota (bar)")
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    city_count = df_filtered["City"].value_counts()
    bars1 = ax1.bar(
        city_count.index,
        city_count.values,
        color=PALETTE[:len(city_count)],
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
    )
    # Label angka di atas tiap batang
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 1.5,
            str(int(h)), ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    ax1.set_xlabel("Kota", fontsize=11)
    ax1.set_ylabel("Jumlah Destinasi", fontsize=11)
    ax1.set_title("Jumlah Destinasi Wisata per Kota", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, city_count.max() * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.divider()

    # Grafik 2 : plt.scatter() – Rating vs Harga per Kategori 
    st.subheader("Grafik 2 – Hubungan Rating vs Harga Tiket per Kategori (scatter)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    categories = df_filtered["Category"].unique()
    for i, cat in enumerate(categories):
        subset = df_filtered[df_filtered["Category"] == cat]
        ax2.scatter(
            subset["Price"],
            subset["Rating"],
            label=cat,
            alpha=0.72,
            s=65,
            color=PALETTE[i % len(PALETTE)],
            edgecolors="white",
            linewidths=0.4,
        )
    ax2.set_xlabel("Harga Tiket (Rp)", fontsize=11)
    ax2.set_ylabel("Rating", fontsize=11)
    ax2.set_title("Hubungan Rating vs Harga Tiket per Kategori", fontsize=13, fontweight="bold")
    # Format sumbu X menjadi Rp
    ax2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"Rp {int(x):,}")
    )
    plt.xticks(rotation=15, ha="right")
    ax2.legend(title="Kategori", fontsize=9, loc="lower right")
    ax2.grid(linestyle="--", alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    #  Grafik 3 : plt.plot() – Rata-rata Rating per Kategori (Line) 
    st.subheader("Grafik 3 – Rata-rata Rating per Kategori Wisata (plot)")
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    avg_rating = (
        df_filtered.groupby("Category")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ax3.plot(
        avg_rating["Category"],
        avg_rating["Rating"],
        marker="o",
        markersize=9,
        linewidth=2.2,
        color=PALETTE[0],
        markerfacecolor=PALETTE[2],
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    # Anotasi nilai di tiap titik
    for _, row in avg_rating.iterrows():
        ax3.annotate(
            f"{row['Rating']:.2f}",
            (row["Category"], row["Rating"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=PALETTE[0],
        )
    ax3.set_xlabel("Kategori", fontsize=11)
    ax3.set_ylabel("Rata-rata Rating", fontsize=11)
    ax3.set_title("Rata-rata Rating per Kategori Wisata", fontsize=13, fontweight="bold")
    ax3.set_ylim(avg_rating["Rating"].min() - 0.1, avg_rating["Rating"].max() + 0.2)
    plt.xticks(rotation=18, ha="right")
    ax3.grid(linestyle="--", alpha=0.35)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()

    #  Grafik 4 : plt.pie() – Proporsi Kategori Wisata 
    st.subheader("Grafik 4 – Proporsi Jumlah Destinasi per Kategori (pie)")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    cat_count = df_filtered["Category"].value_counts()
    wedges, texts, autotexts = ax4.pie(
        cat_count.values,
        labels=cat_count.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(cat_count)],
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax4.set_title("Proporsi Jumlah Destinasi per Kategori", fontsize=13, fontweight="bold")
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.divider()

    #  Grafik 5 : plt.bar() horizontal – Top 10 SAW (setelah dihitung) 
    st.subheader("Grafik 5 – Top 10 Rekomendasi Nilai SAW (bar horizontal)")
    if "df_result" in st.session_state:
        df_res = st.session_state["df_result"]
        top10  = df_res.head(10).copy()

        fig5, ax5 = plt.subplots(figsize=(10, 6))
        colors5 = [PALETTE[0]] + [PALETTE[1]] * (len(top10) - 1)
        bars5 = ax5.barh(
            top10["Place_Name"].values[::-1],
            top10["Nilai_SAW"].values[::-1],
            color=colors5[::-1],
            edgecolor="white",
            height=0.6,
        )
        # Label nilai di ujung batang
        for bar in bars5:
            w = bar.get_width()
            ax5.text(
                w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=9
            )
        ax5.set_xlabel("Nilai SAW", fontsize=11)
        ax5.set_title("Top 10 Rekomendasi Destinasi Wisata (Nilai SAW)", fontsize=13, fontweight="bold")
        ax5.set_xlim(0, top10["Nilai_SAW"].max() * 1.14)
        ax5.grid(axis="x", linestyle="--", alpha=0.35)
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)
    else:
        st.info("Jalankan perhitungan SAW terlebih dahulu di halaman Hitung SPK untuk menampilkan grafik ini.")

# HALAMAN : PROFIL KELOMPOK
elif halaman == "Profil Kelompok":
    st.title("Profil Kelompok")

    st.markdown(
        """
        ### Informasi Proyek
        | | |
        |--|--|
        | **Mata Kuliah** | Sistem dan Cerdas Pendukung Keputusan (SCPK) |
        | **Tahun Ajaran** | 2025 / 2026 |
        | **Metode SPK**  | Simple Additive Weighting (SAW) |
        | **Dataset**     | Indonesia Tourism Destination (437 baris) |
        | **Sumber Data** | Kaggle – Indonesia Tourism Destination |

        ---

        ### Anggota Kelompok
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Anggota 1**
            - **Nama :** *Bisma Putra Pangestu*
            - **NIM  :** *123240217*
            """
        )
    with col2:
        st.markdown(
            """
            **Anggota 2**
            - **Nama :** *Muhammad Syawal Azzami*
            - **NIM  :** *123240224*
            """
        )

    st.divider()