import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="#MensEerst Matrix-Scanner", page_icon="🧼", layout="centered"
)

# Taalkeuze in het zijmenu
taal = st.sidebar.radio("🌐 Taal / Language:", ["Nederlands", "English"])

TEKSTEN = {
    "Nederlands": {
        "titel": "🧼 #MensEerst Matrix-Scanner",
        "intro": """
**Ontworpen voor de Verificatie van Dr. Jin Shanmu's Medische Baseline**  
Deze app gebruikt een deterministische 3600-hoeken-discretisatie via het Hermitische deel om de Crouzeix-conjectuur te visualiseren.

🚨 **BELANGRIJK:** Dit model is wiskundig stabiel, maar dient expliciet in de buitenwereld (klinische praktijk) getest te worden op echte apparatuur en patiëntendata om de operationele veiligheid te valideren.
""",
        "bediening": "🛠️ Bediening",
        "kies_opstelling": "Kies Matrix-Opstelling:",
        "opties": [
            "De Getemde Pestkop (Sharp Case)",
            "Handmatige 3x3 Input Modus",
        ],
        "doelwit_titel": "📊 Doelwit: 3x3 Nilpotente Verschuivingsmatrix",
        "handmatig_titel": "✍️ Handmatige Input (3x3 Complex Matrix)",
        "scan_knop": "🚀 Voer Deterministische Scan Executie Uit",
        "metric_label": "🏆 Gecorrigeerde Crouzeix Ratio",
        "succes": (
            "✅ Grens geverifieerd: Ratio respecteert strikt de theoretische"
            " limiet van 2."
        ),
        "fout": "🚨 Uitzondering op de grens gedetecteerd.",
        "grens_label": "Grens W(A)",
        "eigenwaarden_label": "Eigenwaarden",
        "download_knop": "📥 Download Grafiek als Afbeelding (PNG)",
    },
    "English": {
        "titel": "🧼 #HumanFirst Matrix-Scanner",
        "intro": """
**Designed for the Verification of Dr. Jin Shanmu's Medical Baseline**  
This app uses a deterministic 3600-angle discretization via the Hermitian part to visualize the Crouzeix conjecture.

🚨 **IMPORTANT:** This model is mathematically stable, but must be explicitly tested in the real world (clinical practice) on actual equipment and patient data to validate operational safety.
""",
        "bediening": "🛠️ Controls",
        "kies_opstelling": "Select Matrix Setup:",
        "opties": ["The Tamed Bully (Sharp Case)", "Manual 3x3 Input Mode"],
        "doelwit_titel": "📊 Target: 3x3 Nilpotent Shift Matrix",
        "handmatig_titel": "✍️ Manual Input (3x3 Complex Matrix)",
        "scan_knop": "🚀 Execute Deterministic Scan",
        "metric_label": "🏆 Corrected Crouzeix Ratio",
        "succes": (
            "✅ Bound verified: Ratio strictly respects the theoretical limit"
            " of 2."
        ),
        "fout": "🚨 Exception to the bound detected.",
        "grens_label": "Boundary W(A)",
        "eigenwaarden_label": "Eigenvalues",
        "download_knop": "📥 Download Chart as Image (PNG)",
    },
}

t = TEKSTEN[taal]

st.title(t["titel"])
st.markdown(t["intro"])

st.sidebar.header(t["bediening"])
matrix_type = st.sidebar.selectbox(t["kies_opstelling"], t["opties"])


def bereken_exacte_crouzeix_ratio(A, p_coefficients):
  p_A = np.zeros_like(A, dtype=complex)
  for i, coef in enumerate(p_coefficients):
    p_A += coef * np.linalg.matrix_power(A, i)
  matrix_norm = np.linalg.norm(p_A, ord=2)

  hoeken = np.linspace(0, 2 * np.pi, 3600)
  max_p_z = 0.0
  fov_punten = []

  for theta in hoeken:
    A_hermitisch = (
        np.exp(-1j * theta) * A + (np.exp(-1j * theta) * A).conj().T
    ) / 2.0
    _, eigenvectoren = np.linalg.eigh(A_hermitisch)
    v = eigenvectoren[:, -1]
    z = np.dot(v.conj().T, np.dot(A, v))
    fov_punten.append(z)

    p_z = complex(0)
    for i, coef in enumerate(p_coefficients):
      p_z += coef * (z**i)

    if abs(p_z) > max_p_z:
      max_p_z = abs(p_z)

  ratio = matrix_norm / max_p_z if max_p_z != 0 else 0
  return ratio, fov_punten, max_p_z, matrix_norm


if matrix_type == t["opties"][0]:
  A = np.array([[0, 2, 0], [0, 0, 2], [0, 0, 0]], dtype=complex)
  st.subheader(t["doelwit_titel"])
  st.write(np.real(A))
else:
  st.subheader(t["handmatig_titel"])
  col1, col2, col3 = st.columns(3)
  with col1:
    m00, m10, m20 = (
        st.number_input("R1K1", value=0.0),
        st.number_input("R2K1", value=0.0),
        st.number_input("R3K1", value=0.0),
    )
  with col2:
    m01, m11, m21 = (
        st.number_input("R1K2", value=2.0),
        st.number_input("R2K2", value=0.0),
        st.number_input("R3K2", value=0.0),
    )
  with col3:
    m02, m12, m22 = (
        st.number_input("R1K3", value=0.0),
        st.number_input("R2K3", value=2.0),
        st.number_input("R3K3", value=0.0),
    )
  A = np.array(
      [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]], dtype=complex
  )

p_coef = [0, 0, 1]

if st.button(t["scan_knop"]):
  ratio, fov_punten, max_p_z, matrix_norm = bereken_exacte_crouzeix_ratio(
      A, p_coef
  )
  st.write("---")
  st.metric(label=t["metric_label"], value=f"{ratio:.6f}")

  if ratio <= 2.00005:
    st.success(t["succes"])
  else:
    st.error(t["fout"])

  fig, ax = plt.subplots()
  x_p = [np.real(p) for p in fov_punten] + [np.real(fov_punten[0])]
  y_p = [np.imag(p) for p in fov_punten] + [np.imag(fov_punten[0])]
  ax.plot(x_p, y_p, label=t["grens_label"], color="#2ca02c", linewidth=2)
  ax.fill(x_p, y_p, alpha=0.1, color="#2ca02c")
  eig = np.linalg.eigvals(A)
  ax.scatter(
      np.real(eig),
      np.imag(eig),
      color="red",
      zorder=5,
      label=t["eigenwaarden_label"],
  )
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.set_aspect("equal", "box")
  ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
  st.pyplot(fig)

  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
  st.download_button(
      label=t["download_knop"],
      data=buf.getvalue(),
      file_name="crouzeix_matrix_scan.png",
      mime="image/png",
  )
