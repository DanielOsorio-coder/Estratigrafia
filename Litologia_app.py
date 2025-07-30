
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import io

st.set_page_config(layout="wide")

st.title("Generador de Columnas Estratigráficas")
st.write("""
Autor: Daniel Osorio Álvarez (dosorioalv@gmail.com)
.Elige la Unidad desde la lista desplegable (Monolito, Arena, etc). El patrón y color se asignarán automáticamente según la leyenda.
""")

# --- Simbología por defecto ---
leyenda_default = [
    {'unidad': 'Monolito', 'uh': '', 'color': '#6fa8dc', 'hatch': ''},
    {'unidad': 'Sello sanitario', 'uh': '', 'color': '#313131', 'hatch': ''},
    {'unidad': 'Bentonita Pelet', 'uh': '', 'color': '#d9ead3', 'hatch': '...'},
    {'unidad': 'Bentonita Polvo', 'uh': '', 'color': '#d9ead3', 'hatch': '////'},
    {'unidad': 'Arena', 'uh': '', 'color': '#ffe599', 'hatch': '---'},
    {'unidad': 'Gravas', 'uh': '', 'color': '#ffe599', 'hatch': '...'},
    {'unidad': 'Lechada', 'uh': '', 'color': '#f4cccc', 'hatch': '|||'},
    {'unidad': 'Esteril', 'uh': '', 'color': '#eeeeee', 'hatch': 'xx'},
    {'unidad': 'Derrumbes', 'uh': '', 'color': '#b7b7b7', 'hatch': '///...'},
    {'unidad': 'Basamento', 'uh': '1a', 'color': '#a9746e', 'hatch': 'xxxx'},
    {'unidad': 'Rocas sedimentarias', 'uh': '1b', 'color': '#f6b26b', 'hatch': '.'},
    {'unidad': 'Zona de falla', 'uh': '1c', 'color': '#b6d7a8', 'hatch': '////'},
    {'unidad': 'Ignimbrita Huasco', 'uh': '2', 'color': '#fff2cc', 'hatch': '...'},
    {'unidad': 'Conjunto Volcánico Antiguo', 'uh': '3', 'color': '#c9daf8', 'hatch': '---'},
    {'unidad': 'Ignimbrita Ujina', 'uh': '4', 'color': '#f9cb9c', 'hatch': '.'},
    {'unidad': 'Subunidad Volcánico-Sedimentaria Brechosa', 'uh': '5a', 'color': '#d5a6bd', 'hatch': 'xxx'},
    {'unidad': 'Subunidad Evaporítica Profunda', 'uh': '5b', 'color': '#d9d2e9', 'hatch': '||'},
    {'unidad': 'Subunidad Volcánico-Sedimentaria Superior', 'uh': '5c', 'color': '#d0e0e3', 'hatch': '\\\\'},
    {'unidad': 'Conjunto Volcánico Moderno', 'uh': '6', 'color': '#b49154', 'hatch': '...'},
    {'unidad': 'Ignimbrita Pastillo', 'uh': '7', 'color': '#ffffff', 'hatch': ''},
    {'unidad': 'Depósitos Sedimentarias Terciarios', 'uh': '8a', 'color': '#fce5cd', 'hatch': '//'},
    {'unidad': 'Depósitos Evaporíticos', 'uh': '8b', 'color': '#b4a7d6', 'hatch': ''},
    {'unidad': 'Relleno Sedimentario', 'uh': '8c', 'color': '#b7b7b7', 'hatch': '.'},
]

# --- Permitir customización de simbología (color, patrón, nombre, agregar/eliminar) ---
if "leyenda_custom" not in st.session_state:
    st.session_state.leyenda_custom = [d.copy() for d in leyenda_default]

with st.expander("Editar simbología de unidades (color, patrón y nombre)", expanded=False):
    st.write("Puedes agregar, eliminar o editar el nombre, color y patrón de cada unidad:")

    # Para manejar unidades a eliminar
    to_delete = []
    for i, simb in enumerate(st.session_state.leyenda_custom):
        cols = st.columns([2, 2, 1, 1, 0.6])
        with cols[0]:
            simb['unidad'] = st.text_input(f"Nombre de la unidad {i+1}", value=simb['unidad'], key=f"unidadname_{i}")
        with cols[1]:
            simb['color'] = st.color_picker(f"Color para {simb['unidad']}", value=simb['color'], key=f"colpick_{i}")
        with cols[2]:
            simb['hatch'] = st.text_input(f"Patrón (hatch) para {simb['unidad']}", value=simb['hatch'], key=f"hatch_{i}")
        with cols[4]:
            if st.button("🗑️", key=f"del_unidad_{i}"):
                to_delete.append(i)
    # Borrar luego de iterar (al revés para no desordenar índices)
    for idx in sorted(to_delete, reverse=True):
        del st.session_state.leyenda_custom[idx]

    # Agregar unidad
    if st.button("➕ Agregar nueva unidad"):
        st.session_state.leyenda_custom.append(
            {'unidad': f"Nueva Unidad {len(st.session_state.leyenda_custom)+1}", 'uh': '', 'color': '#ffffff', 'hatch': ''}
        )

    ccol1, ccol2 = st.columns([1,1])
    with ccol1:
        if st.button("🔄 Restaurar simbología por defecto"):
            st.session_state.leyenda_custom = [d.copy() for d in leyenda_default]
    with ccol2:
        st.info("Patrones (hatch) típicos: `/`, `\\`, `x`, `xx`, `|`, `-`, `+`, `.`, `...`, `//`, `|||`, etc. (puedes combinarlos)")

# --- Usar simbología customizada ---
leyenda_actual = st.session_state.leyenda_custom
unidades_lista = [d['unidad'] for d in leyenda_actual]
leyenda_lookup = {d['unidad']: d for d in leyenda_actual}

# --- Sincroniza la columna 'Unidad' con la leyenda (elimina valores no existentes y actualiza cambios de nombre) ---
if "df" in st.session_state:
    # Cambios de nombre: mapping desde default a custom
    old_names = [d['unidad'] for d in leyenda_default]
    new_names = [d['unidad'] for d in leyenda_actual[:len(leyenda_default)]]
    mapping = {old: new for old, new in zip(old_names, new_names)}
    st.session_state.df["Unidad"] = st.session_state.df["Unidad"].replace(mapping)
    # Si hay unidades eliminadas, deja en blanco las que ya no existen
    st.session_state.df.loc[~st.session_state.df["Unidad"].isin(unidades_lista), "Unidad"] = ""

# --- Leyenda visual arriba del editor ---
st.markdown("### Leyenda de patrones para columna Unidad:")
fig_leyenda, axl = plt.subplots(figsize=(7.2, 1.5))
axl.axis('off')
for i, simb in enumerate(leyenda_actual):
    axl.add_patch(
        mpatches.Rectangle((i, 0.2), 0.88, 0.5, facecolor=simb['color'],
                           hatch=simb['hatch'], edgecolor='black', linewidth=1.2)
    )
    axl.text(i+0.44, 0.75, f"`{simb['hatch']}`", ha='center', va='bottom', fontsize=5, color='dimgray', family='monospace')
    axl.text(i+0.44, 0.13, simb['unidad'], ha='center', va='top', fontsize=3.8, rotation=90)
axl.set_xlim(0, len(leyenda_actual))
axl.set_ylim(0, 1)
st.pyplot(fig_leyenda)

# --- DataFrame editable SOLO con Unidad como selectbox ---
if "df" not in st.session_state:
    data = {
        "Profundidad_sup": [0, 20, 40, 60, 80, 110, 130, 150, 180],
        "Profundidad_inf": [20, 40, 60, 80, 110, 130, 150, 180, 210],
        "Litologia": [
            "Toba cristalina grises a rosáceas, fragmentos grava a arena, sin arcilla.",
            "Brecha de falla pardo-rojiza, meteorizada, con arcilla.",
            "Toba morada-gris con oxidación, bajo contenido de arcilla.",
            "Intrusivo gris, identificado como dique con vetas de Qz.",
            "Toba morada-gris alterada (Qz, Ser), sin arcilla, con intrusivos.",
            "Brecha de falla pardo-rojiza, meteorizada, con arcilla.",
            "Toba morada-gris alterada, intrusivos cortando la unidad.",
            "Fragmentos de roca arena, zona de falla.",
            "Estéril y derrumbes con gravas."
        ],
        "Unidad": ['Monolito', 'Sello sanitario', 'Bentonita Pelet', 'Bentonita Polvo', 'Arena',
                   'Gravas', 'Lechada', 'Esteril', 'Derrumbes'],
        "UH": ['UH-4b', 'UH-1c', 'UH-4b', 'UH-1a', 'UH-4b', 'UH-1a', 'UH-4b', 'UH-1c', 'UH-1c'],
    }
    st.session_state.df = pd.DataFrame(data)

# --- Añadir y eliminar filas ---
c1, c2 = st.columns([1,1.2])
with c1:
    if st.button("➕ Añadir fila"):
        df_add = st.session_state.df.copy()
        new_row = {
            "Profundidad_sup": 0, "Profundidad_inf": 10,
            "Litologia": "", "Unidad": unidades_lista[0] if len(unidades_lista) > 0 else "", "UH": ""
        }
        st.session_state.df = pd.concat([df_add, pd.DataFrame([new_row])], ignore_index=True)
with c2:
    if st.button("🗑️ Eliminar última fila") and len(st.session_state.df) > 1:
        st.session_state.df = st.session_state.df.iloc[:-1].reset_index(drop=True)

# --- Edición de datos ---
st.write("**Edita los datos:**")
df_input = st.session_state.df.copy()
for idx in df_input.index:
    cols = st.columns([1, 1.1, 2, 1.1])
    with cols[0]:
        df_input.at[idx, "Profundidad_sup"] = st.number_input(
            f"Prof. sup. fila {idx+1}", value=float(df_input.at[idx, "Profundidad_sup"]), key=f"sup_{idx}", step=1.0
        )
    with cols[1]:
        df_input.at[idx, "Profundidad_inf"] = st.number_input(
            f"Prof. inf. fila {idx+1}", value=float(df_input.at[idx, "Profundidad_inf"]), key=f"inf_{idx}", step=1.0
        )
    with cols[2]:
        df_input.at[idx, "Litologia"] = st.text_input(
            f"Litología fila {idx+1}", value=df_input.at[idx, "Litologia"], key=f"lito_{idx}"
        )
    with cols[3]:
        # Si la unidad actual está eliminada, la deja vacía
        current = df_input.at[idx, "Unidad"]
        if current not in unidades_lista:
            current = ""
        df_input.at[idx, "Unidad"] = st.selectbox(
            f"Unidad fila {idx+1}", options=[""] + unidades_lista, index=([""] + unidades_lista).index(current), key=f"unidad_{idx}"
        )
    df_input.at[idx, "UH"] = st.text_input(
        f"UH fila {idx+1}", value=df_input.at[idx, "UH"], key=f"uh_{idx}"
    )
    st.markdown("---")

df_input["Color"] = df_input["Unidad"].map(lambda x: leyenda_lookup[x]["color"] if x in leyenda_lookup else "#ffffff")
df_input["Hatch"] = df_input["Unidad"].map(lambda x: leyenda_lookup[x]["hatch"] if x in leyenda_lookup else "")
st.session_state.df = df_input

# --- Plot matplotlib ---
prof_max = df_input["Profundidad_inf"].max()
prof_min = df_input["Profundidad_sup"].min()

fig = plt.figure(figsize=(9.5, 10))
gs = fig.add_gridspec(2, 5, height_ratios=[14, 1.5], width_ratios=[2.6, 0.4, 1, 1.5, 1.1], hspace=0.08, wspace=0.13)
ax_lit = fig.add_subplot(gs[0, 0])
ax0 = fig.add_subplot(gs[0, 2], sharey=ax_lit)
ax1 = fig.add_subplot(gs[0, 3], sharey=ax_lit)
ax_text = fig.add_subplot(gs[0, 4], sharey=ax_lit)
ax_leg = fig.add_subplot(gs[1, :])

ax_lit.set_ylim(prof_max, prof_min)
ax_lit.set_xlim(0, 1)
ax_lit.axis('off')
ax_lit.set_title("Descripción Litología", fontsize=10, weight="bold", pad=16)
for i, row in df_input.iterrows():
    height = row["Profundidad_inf"] - row["Profundidad_sup"]
    fontsize = max(8, min(13, height * 0.17))
    wrapper = textwrap.TextWrapper(width=40)
    text_wrapped = "\n".join(wrapper.wrap(str(row["Litologia"])))
    max_lines = int(height // 6.5)
    lines = text_wrapped.split('\n')
    if len(lines) > max_lines:
        text_wrapped = '\n'.join(lines[:max_lines]) + '\n...'
    ax_lit.add_patch(
        mpatches.Rectangle(
            (0.04, row["Profundidad_sup"]), 0.92, height,
            fill=False, edgecolor='black', linewidth=1.1, zorder=1)
    )
    ax_lit.text(
        0.5, (row["Profundidad_sup"] + row["Profundidad_inf"]) / 2,
        text_wrapped, va="center", ha="center",
        fontsize=fontsize, wrap=True, zorder=2, clip_on=True,
        bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", linewidth=0)
    )

ax0.set_ylim(prof_max, prof_min)
ax0.set_xlim(0, 1)
ax0.set_yticks([])
ax0.set_xticks([])
ax0.axis('off')
ax0.set_title("Profundidad", fontsize=10, weight="bold", pad=16)
for y in range(0, int(prof_max)+2, 2):
    if y % 10 == 0:
        ax0.plot([0.68, 1.0], [y, y], color="steelblue", lw=1.7)
        ax0.text(0.63, y, str(y), ha="right", va="center", fontsize=10, color="black")
    elif y % 5 == 0:
        ax0.plot([0.80, 1.0], [y, y], color="steelblue", lw=1.1)
    else:
        ax0.plot([0.89, 1.0], [y, y], color="steelblue", lw=0.9)
ax0.plot([1.0, 1.0], [prof_min, prof_max], color="steelblue", lw=2)

ax1.set_ylim(prof_max, prof_min)
ax1.set_xlim(0, 1)
ax1.axis("off")
ax1.set_title("Unidad Hidrogeológica", fontsize=10, weight="bold", pad=16)
for i, row in df_input.iterrows():
    ax1.barh(
        y=(row["Profundidad_sup"] + row["Profundidad_inf"]) / 2,
        width=1,
        height=(row["Profundidad_inf"] - row["Profundidad_sup"]),
        left=0,
        color=row["Color"],
        edgecolor="black",
        hatch=row["Hatch"],
        linewidth=1.2,
        zorder=1
    )

ax_text.set_ylim(prof_max, prof_min)
ax_text.set_xlim(0, 1)
ax_text.axis("off")
for i, row in df_input.iterrows():
    ax_text.text(
        0.01, (row["Profundidad_sup"] + row["Profundidad_inf"]) / 2,
        row["UH"], va="center", ha="left", fontsize=9, fontweight="bold", color="black"
    )

ax_leg.axis('off')
ax_leg.set_xlim(0, len(leyenda_actual))
ax_leg.set_ylim(0, 1.6)
for i, simb in enumerate(leyenda_actual):
    ax_leg.add_patch(
        mpatches.Rectangle((i+0.02, 0.82), 0.75, 0.45, facecolor=simb['color'],
                           hatch=simb['hatch'], edgecolor='black', linewidth=1.1)
    )
    ax_leg.text(i+0.38, 0.78, simb['unidad'], ha='center', va='top', fontsize=6.5, rotation=90, weight='bold')
ax_leg.set_title("Leyenda", fontsize=10, loc='left', pad=7, rotation=0, weight='bold')

plt.tight_layout(rect=[0, 0.04, 1, 1])
st.pyplot(fig)

# ==== DESCARGA COMO PNG Y SVG ====
# Guarda la figura en buffers de memoria
png_buffer = io.BytesIO()
fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
png_buffer.seek(0)

svg_buffer = io.BytesIO()
fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
svg_buffer.seek(0)

st.markdown("### Descargar figura:")

colp, cols = st.columns(2)
with colp:
    st.download_button(
        label="📥 Descargar como PNG",
        data=png_buffer,
        file_name="columna_estratigrafica.png",
        mime="image/png"
    )
with cols:
    st.download_button(
        label="📥 Descargar como SVG",
        data=svg_buffer,
        file_name="columna_estratigrafica.svg",
        mime="image/svg+xml"
    )
