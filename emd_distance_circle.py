import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_error_space(emd_score, hop_length=512, sample_rate=16000):
    """
    Verilen Earth Mover (Euclidean) mesafesi için hata uzayını çizer.

    Args:
        emd_score (float): Modelden gelen ortalama hata puanı (Örn: 3.0).
        hop_length (int): Frame başına sample sayısı (Zaman hesabı için).
        sample_rate (int): Örnekleme hızı.
    """

    # 1 Frame kaç ms? (512 / 16000 = 0.032s = 32ms)
    ms_per_frame = (hop_length / sample_rate) * 1000

    # --- 1. ÇEMBER VERİSİ (Sınır) ---
    theta = np.linspace(0, 2 * np.pi, 200)
    x_circle = emd_score * np.cos(theta)
    y_circle = emd_score * np.sin(theta)

    # --- 2. OLASI TAM SAYI HATALAR (Grid Points) ---
    # Pitch ve Time discrete (tam sayı) olduğu için sadece tam sayı noktaları işaretleyelim.
    limit = int(np.ceil(emd_score))
    possible_points_x = []
    possible_points_y = []

    # Grid taraması
    for x in range(-limit, limit + 1):
        for y in range(-limit, limit + 1):
            # Eğer nokta çemberin içindeyse veya tam üstündeyse
            dist = np.sqrt(x**2 + y**2)
            if dist <= emd_score:
                possible_points_x.append(x)
                possible_points_y.append(y)

    # --- 3. ÇİZİM ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Merkez (Doğru Nota)
    ax.scatter(
        0,
        0,
        color="green",
        s=200,
        label="Hedef (Doğru Nota)",
        zorder=10,
        edgecolors="black",
    )

    # Hata Çemberi
    ax.plot(
        x_circle,
        y_circle,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Ortalama Hata Sınırı ({emd_score:.2f} px)",
    )
    ax.fill(x_circle, y_circle, color="red", alpha=0.1)  # Alanı boya

    # Olası Hatalı Kombinasyonlar
    ax.scatter(
        possible_points_x,
        possible_points_y,
        color="blue",
        alpha=0.6,
        s=50,
        label="Olası Hatalı Tahminler",
    )

    # --- 4. ETİKETLER VE DETAYLAR ---

    # Eksenleri Güzelleştir
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Eşit ölçek (Çember daire gibi görünsün diye)
    ax.set_aspect("equal", adjustable="box")

    # Eksen Yazıları
    ax.set_xlabel(
        f"Zaman Hatası (Frame)\n(1 Frame ≈ {ms_per_frame:.1f} ms)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Nota Hatası (Semitone)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Hata Uzayı Görselleştirmesi\nEarth Mover Distance = {emd_score}", fontsize=14
    )

    # Sınırları ayarla
    buffer = 1
    ax.set_xlim(-emd_score - buffer, emd_score + buffer)
    ax.set_ylim(-emd_score - buffer, emd_score + buffer)

    # Önemli Noktalara Annotation Ekle
    # Örnek: Sadece Zaman Hatası
    ax.annotate(
        f"Sadece Zaman Kayması\n({emd_score} frame = {emd_score * ms_per_frame:.0f} ms)",
        xy=(emd_score, 0),
        xytext=(emd_score + 0.5, 0.5),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    # Örnek: Sadece Nota Hatası
    ax.annotate(
        f"Sadece Pitch Hatası\n({emd_score} yarım ses)",
        xy=(0, emd_score),
        xytext=(0.5, emd_score + 0.5),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    plt.legend(loc="lower right")
    plt.tight_layout()

    # Kaydet ve Göster
    filename = f"error_circle_emd_{emd_score}.png"
    plt.savefig(filename)
    print(f"Grafik kaydedildi: {filename}")
    plt.show()


# --- TEST KULLANIMI ---
# Modelinin verdiği ortalama distance'ı buraya gir:
ORNEK_EMD = 3.0
plot_error_space(ORNEK_EMD)
