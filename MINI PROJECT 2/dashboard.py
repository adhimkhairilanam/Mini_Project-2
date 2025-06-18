import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dashboard Analisis Kepribadian",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk dark mode dan styling yang lebih baik
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #444;
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #444;
    }
    .notes-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #444;
    }
    .data-quality-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #444;
    }
    .stDataFrame {
        background-color: #1e1e1e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    """Load dan bersihkan dataset personality dengan handling missing values"""
    try:
        # Load data dengan skip baris deskripsi jika ada
        df = pd.read_csv('personality_dataset.csv', skiprows=1)
        
        # Bersihkan nama kolom
        df.columns = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                      'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                      'Post_frequency', 'Personality']
        
        # Konversi kolom numerik dan handle missing values
        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                       'Friends_circle_size', 'Post_frequency']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        
        # Handle missing values untuk kolom kategorikal
        df['Stage_fear'].fillna(df['Stage_fear'].mode()[0], inplace=True)
        df['Drained_after_socializing'].fillna(df['Drained_after_socializing'].mode()[0], inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Dashboard Analisis Kepribadian</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load dan bersihkan data
    df = load_and_clean_data()
    
    if df is None:
        st.error("Gagal memuat dataset. Pastikan file 'personality_dataset.csv' tersedia.")
        return
    
    # Sidebar
    st.sidebar.title("📊Dashboard")
    st.sidebar.markdown("---")
    
    # Dataset overview
    st.sidebar.subheader("📋 Ringkasan Dataset")
    st.sidebar.info(f"**Total Record:** {len(df)}")
    st.sidebar.info(f"**Jumlah Fitur:** {len(df.columns)}")
    
    # Distribusi kepribadian
    personality_counts = df['Personality'].value_counts()
    st.sidebar.markdown("**Distribusi Kepribadian:**")
    for personality, count in personality_counts.items():
        st.sidebar.info(f"• {personality}: {count}")
    
    # Filter options
    st.sidebar.subheader("🔍 Filter Data")
    selected_personality = st.sidebar.multiselect(
        "Pilih Tipe Kepribadian",
        options=df['Personality'].unique(),
        default=df['Personality'].unique()
    )
    
    # Filter data
    filtered_df = df[df['Personality'].isin(selected_personality)]
    
    # Definisi kolom numerik
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
    
    # Main content area - Metrics
    st.subheader("📊 Metrik Utama")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Partisipan", len(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_alone = filtered_df['Time_spent_Alone'].mean()
        st.metric("Rata-rata Waktu Sendirian", f"{avg_alone:.1f} jam")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_friends = filtered_df['Friends_circle_size'].mean()
        st.metric("Rata-rata Lingkaran Pertemanan", f"{avg_friends:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_social = filtered_df['Social_event_attendance'].mean()
        st.metric("Rata-rata Kehadiran Sosial", f"{avg_social:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== VISUALISASI 1: DISTRIBUSI KEPRIBADIAN =====
    st.subheader("🎯 Visualisasi 1: Distribusi Tipe Kepribadian")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        personality_counts = filtered_df['Personality'].value_counts()
        
        # Warna yang bagus untuk dark mode
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=personality_counts.index,
            values=personality_counts.values,
            hole=0.4,
            marker=dict(colors=colors[:len(personality_counts)], line=dict(color='#000000', width=2)),
            textinfo='label+percent+value',
            textfont=dict(size=14, color='white'),
            showlegend=True
        )])
        
        fig_pie.update_layout(
            title={
                'text': "Distribusi Tipe Kepribadian dalam Dataset",
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            height=450,
            font=dict(size=14, color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='white'))
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown('<div class="notes-box">', unsafe_allow_html=True)
        st.markdown("**📝 Tujuan Visualisasi:**")
        st.markdown("Menunjukkan komposisi dan proporsi setiap tipe kepribadian dalam dataset untuk memahami representasi sampel penelitian.")
        
        st.markdown("**💡 Insight yang Ditunjukkan:**")
        total_count = len(filtered_df)
        for personality, count in personality_counts.items():
            percentage = (count/total_count)*100
            st.markdown(f"• **{personality}:** {count} orang ({percentage:.1f}%)")
        
        if len(personality_counts) > 0:
            dominant_type = personality_counts.index[0]
            st.markdown(f"• **Tipe Dominan:** {dominant_type}")
        
        st.markdown("**🎯 Manfaat:** Memvalidasi keseimbangan dataset dan menentukan strategi analisis lanjutan.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== VISUALISASI 2: SCATTER PLOT HUBUNGAN WAKTU SENDIRI VS AKTIVITAS SOSIAL =====
    st.subheader("📈 Visualisasi 2: Hubungan Waktu Sendiri vs Aktivitas Sosial")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_scatter = px.scatter(
            filtered_df, 
            x='Time_spent_Alone', 
            y='Social_event_attendance',
            color='Personality',
            size='Friends_circle_size',
            hover_data=['Going_outside', 'Post_frequency'],
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            title="Korelasi: Waktu Sendiri vs Kehadiran Acara Sosial",
            labels={
                'Time_spent_Alone': 'Waktu Sendiri (jam/hari)',
                'Social_event_attendance': 'Kehadiran Acara Sosial (skala 0-10)',
                'Friends_circle_size': 'Ukuran Lingkaran Pertemanan'
            }
        )
        
        fig_scatter.update_layout(
            height=500,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='gray', gridwidth=0.5),
            yaxis=dict(gridcolor='gray', gridwidth=0.5)
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.markdown('<div class="notes-box">', unsafe_allow_html=True)
        st.markdown("**📝 Tujuan Visualisasi:**")
        st.markdown("Menganalisis hubungan antara preferensi waktu sendiri dengan tingkat partisipasi dalam aktivitas sosial.")
        
        # Hitung korelasi
        correlation = filtered_df['Time_spent_Alone'].corr(filtered_df['Social_event_attendance'])
        st.markdown(f"**📊 Korelasi:** {correlation:.3f}")
        
        if abs(correlation) > 0.7:
            st.markdown("🔴 **Korelasi Kuat**")
        elif abs(correlation) > 0.3:
            st.markdown("🟡 **Korelasi Sedang**")
        else:
            st.markdown("🟢 **Korelasi Lemah**")
        
        st.markdown("**💡 Insight yang Ditunjukkan:**")
        st.markdown("• Pola hubungan negatif antara waktu sendiri dan aktivitas sosial")
        st.markdown("• Clustering berdasarkan tipe kepribadian")
        st.markdown("• Variasi ukuran lingkaran pertemanan")
        
        st.markdown("**🎯 Manfaat:** Memahami trade-off antara kebutuhan privasi dan sosialisasi.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    # ===== VISUALISASI 3: RADAR CHART PROFIL KEPRIBADIAN =====
    st.subheader("🕸️ Visualisasi 3: Profil Radar Karakteristik per Tipe Kepribadian")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Hitung rata-rata per kepribadian
        personality_profiles = {}
        for personality in filtered_df['Personality'].unique():
            if personality in selected_personality:
                subset = filtered_df[filtered_df['Personality'] == personality]
                profile = [
                    subset['Time_spent_Alone'].mean(),
                    subset['Social_event_attendance'].mean(),
                    subset['Going_outside'].mean(),
                    subset['Friends_circle_size'].mean(),
                    subset['Post_frequency'].mean()
                ]
                personality_profiles[personality] = profile
        
        # Label untuk radar chart
        categories = ['Waktu Sendiri', 'Acara Sosial', 'Keluar Rumah', 'Lingkaran Teman', 'Posting Sosmed']
        
        fig_radar = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        color_idx = 0
        
        for personality, profile in personality_profiles.items():
            fig_radar.add_trace(go.Scatterpolar(
                r=profile + [profile[0]],  # Tutup lingkaran
                theta=categories + [categories[0]],
                fill='toself',
                name=personality,
                line_color=colors[color_idx % len(colors)],
                opacity=0.7
            ))
            color_idx += 1
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(profile) for profile in personality_profiles.values()]) * 1.1]
                )),
            showlegend=True,
            title="Profil Karakteristik Rata-rata per Tipe Kepribadian",
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown('<div class="notes-box">', unsafe_allow_html=True)
        st.markdown("**📝 Tujuan Visualisasi:**")
        st.markdown("Menampilkan profil komprehensif setiap tipe kepribadian dalam bentuk radar untuk perbandingan multi-dimensi yang intuitif.")
        
        st.markdown("**💡 Interpretasi Radar:**")
        st.markdown("• **Area lebih besar:** Karakteristik lebih tinggi")
        st.markdown("• **Bentuk polygon:** Pola unik kepribadian")
        st.markdown("• **Overlap area:** Kesamaan traits")
        
        st.markdown("**📊 Profil Kepribadian:**")
        for personality, profile in personality_profiles.items():
            max_trait_idx = np.argmax(profile)
            max_trait = categories[max_trait_idx]
            st.markdown(f"• **{personality}:** Dominan di {max_trait}")
        
        st.markdown("**🎯 Manfaat:** Visualisasi holistik untuk identifikasi pola kepribadian dan pengembangan strategi intervensi personal.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # KESIMPULAN DAN REKOMENDASI
    st.subheader("🔍 Kesimpulan Analisis & Rekomendasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🎯 Insight:**")
        
        # Analisis otomatis berdasarkan data
        personality_types = filtered_df['Personality'].unique()
        st.markdown(f"• **Variasi Kepribadian:** {len(personality_types)} tipe ditemukan")
        
        # Cari karakteristik yang paling membedakan
        max_diff_var = None
        max_diff = 0
        
        for var in numeric_cols:
            means_by_type = filtered_df.groupby('Personality')[var].mean()
            diff = means_by_type.max() - means_by_type.min()
            if diff > max_diff:
                max_diff = diff
                max_diff_var = var
        
        if max_diff_var:
            var_names = {
                'Time_spent_Alone': 'Waktu Sendiri',
                'Social_event_attendance': 'Kehadiran Sosial', 
                'Going_outside': 'Aktivitas Luar',
                'Friends_circle_size': 'Lingkaran Pertemanan',
                'Post_frequency': 'Aktivitas Media Sosial'
            }
            st.markdown(f"• **Pembeda Utama:** {var_names.get(max_diff_var, max_diff_var)}")
            st.markdown(f"• **Rentang Perbedaan:** {max_diff:.2f}")
        
        # Korelasi terkuat
        corr_matrix = filtered_df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)
        max_corr = corr_matrix_masked.abs().max().max()
        
        if not np.isnan(max_corr):
            st.markdown(f"• **Korelasi Terkuat:** {max_corr:.3f}")
        
        st.markdown("• **Kualitas Data:** Excellent (post-cleaning)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**💡 Rekomendasi Aplikasi:**")
        st.markdown("• **HR & Recruitment:** Profiling kandidat berdasarkan fit budaya perusahaan")
        st.markdown("• **Pendidikan:** Personalisasi metode pembelajaran")
        st.markdown("• **Marketing:** Segmentasi pelanggan untuk targeting yang tepat")
        st.markdown("• **Kesehatan Mental:** Deteksi early warning dan intervensi preventif")
        st.markdown("• **Pengembangan Diri:** Rekomendasi program pengembangan personal")
        
        st.markdown("**🔬 Metodologi Lanjutan:**")
        st.markdown("• **Machine Learning:** Klasifikasi otomatis kepribadian")
        st.markdown("• **Cluster Analysis:** Segmentasi berbasis behavior")
        st.markdown("• **Predictive Analytics:** Forecasting perilaku")
        st.markdown("• **A/B Testing:** Validasi intervensi berbasis kepribadian")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Dashboard Analisis Kepribadian - ADHIM KHAIRIL ANAM**")
    st.markdown("**📊 Dataset telah melalui proses data cleaning dan siap untuk analisis mendalam**")

if __name__ == "__main__":
    main()
