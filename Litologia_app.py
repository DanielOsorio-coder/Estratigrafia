import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import io
import numpy as np
from PIL import Image, ImageChops

# === Cropper opcional (con fallback) ===
try:
    from streamlit_cropper import st_cropper
    CROP_AVAILABLE = True
except Exception:
    CROP_AVAILABLE = False

st.set_page_config(layout="wide")
st.title("Generador de Columnas Estratigráficas")
st.write("""
Autor: Daniel Osorio Álvarez (dosorioalv@gmail.com)
.Elige la Unidad desde la lista desplegable (Monolito, Arena, etc). El patrón y color se asignarán automáticamente según la leyenda.
""")

# --- Controles de layout (sidebar) ---
st.sidebar.markdown("### Layout")
col_unidad_ratio = st.sidebar.slider("Ancho columna Unidad", 2.0, 6.0, 4.2, 0.1)
col_uh_ratio     = st.sidebar.slider("Ancho columna UH",     1.0, 4.0, 2.2, 0.1)
fig_width        = st.sidebar.slider("Ancho total (figsize)", 10.0, 20.0, 16.0, 0.5)

st.sidebar.markdown("### Opciones de Contain")
contain_match_width  = st.sidebar.checkbox("Contain: usar ancho del motivo", True)
contain_apply_global = st.sidebar.checkbox("Contain: aplicar ancho a toda la columna", True)
contain_align_label  = st.sidebar.selectbox("Contain: alineación", ["Centrado", "Izquierda", "Derecha"], index=0)
contain_fallback_pct = st.sidebar.slider("Contain: ancho manual si no hay imagen (%)", 20, 100, 70, 1)
_align_map = {"Centrado": "center", "Izquierda": "left", "Derecha": "right"}

# -------------------------
# Utilidades
# -------------------------
def hex_to_rgb(hexcolor: str):
    hexcolor = hexcolor.lstrip('#')
    return tuple(int(hexcolor[i:i+2], 16) for i in (0, 2, 4))

def process_symbol_pil_only(img, bg_rgb=(255, 255, 255), tol=30, symbol_rgb=(0, 0, 0)) -> Image.Image:
    base = img.convert("RGB")
    W, H = base.size
    bg = Image.new("RGB", (W, H), bg_rgb)
    diff = ImageChops.difference(base, bg)
    dist = diff.convert("L")
    alpha = dist.point(lambda v: 0 if v <= tol else 255, mode="L")
    colored = Image.new("RGBA", (W, H), (*symbol_rgb, 255))
    colored.putalpha(alpha)
    return colored  # RGBA

def fill_rect_with_image(
    ax,
    img_rgba: Image.Image,
    x0, y0, width, height,
    scale: float = 1.0,
    zorder: float = 1.0,
    mode: str = "cover",           # "cover" | "fit" | "contain" | "tile"
    align_phase: bool = True,      # para "tile": alinear fase al sistema de pixeles del eje
    align: str = "center",         # "left" | "center" | "right" (si width_override)
    width_override: float = None   # ancho en coords de datos; si None usa 'width'
):
    # --- Ancho efectivo del rectángulo ---
    rect_w = float(width_override) if width_override is not None else float(width)
    rect_w = max(1e-6, min(rect_w, width))  # clamp
    if align == "left":
        x_left = x0
    elif align == "right":
        x_left = x0 + (width - rect_w)
    else:  # center
        x_left = x0 + (width - rect_w) / 2.0

    # Conversión rectángulo a pixeles
    p0 = ax.transData.transform((x_left, y0))
    p1 = ax.transData.transform((x_left + rect_w, y0 + height))
    px_w = max(1, int(abs(p1[0] - p0[0])))
    px_h = max(1, int(abs(p1[1] - p0[1])))

    iw, ih = img_rgba.size

    if mode == "fit":
        final_img = img_rgba.resize((px_w, px_h), Image.NEAREST)

    elif mode == "contain":
        base_w = max(1, int(px_w * scale))
        base_h = max(1, int(px_h * scale))
        s = min(base_w / iw, base_h / ih)
        new_w, new_h = max(1, int(iw * s)), max(1, int(ih * s))
        resized = img_rgba.resize((new_w, new_h), Image.NEAREST)
        canvas = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
        off_x = (base_w - new_w) // 2
        off_y = (base_h - new_h) // 2
        canvas.paste(resized, (off_x, off_y), resized)
        final_img = canvas.resize((px_w, px_h), Image.NEAREST)

    elif mode == "tile":
        tile_w = max(1, int(iw * scale))
        tile_h = max(1, int(ih * scale))
        tile = img_rgba.resize((tile_w, tile_h), Image.NEAREST)
        canvas = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
        off_x = 0
        off_y = 0
        if align_phase:
            phase_x = int(p0[0]) % tile_w
            phase_y = int(p0[1]) % tile_h
            off_x = -phase_x
            off_y = -phase_y
        y = off_y
        while y < px_h:
            x = off_x
            while x < px_w:
                canvas.paste(tile, (x, y), tile)
                x += tile_w
            y += tile_h
        final_img = canvas

    else:  # cover
        base_w = max(1, int(px_w * scale))
        base_h = max(1, int(px_h * scale))
        s = max(base_w / iw, base_h / ih)
        new_w, new_h = max(1, int(iw * s)), max(1, int(ih * s))
        resized = img_rgba.resize((new_w, new_h), Image.NEAREST)
        left   = max(0, (new_w - base_w) // 2)
        top    = max(0, (new_h - base_h) // 2)
        right  = min(new_w, left + base_w)
        bottom = min(new_h, top + base_h)
        cropped = resized.crop((left, top, right, bottom))
        final_img = cropped.resize((px_w, px_h), Image.NEAREST)

    arr = np.asarray(final_img, dtype=np.uint8)
    im = ax.imshow(
        arr,
        extent=[x_left, x_left + rect_w, y0, y0 + height],
        origin='lower',
        interpolation='nearest',
        zorder=zorder,
        aspect='auto'
    )
    im.set_rasterized(True)
    return {"rect_x0": x_left, "rect_width": rect_w, "rect_height": height}

# -------------------------
# Simbología por defecto
# -------------------------
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

# -------------------------
# Estado con campos de imagen
# -------------------------
if "leyenda_custom" not in st.session_state:
    st.session_state.leyenda_custom = [
        {**d, 'img_bytes': None, 'img_scale': 1.0, 'img_mode': 'cover',
         'img_native_w': None, 'img_native_h': None} for d in leyenda_default
    ]

# ===========================================================
# Editor de simbología + recorte desde catálogo
# ===========================================================
with st.expander("Editar simbología (color, patrón y símbolo por imagen con recorte)", expanded=False):
    st.caption("Puedes cargar una **lámina de símbolos** (PNG/JPG) y recortar un rectángulo para usarlo como patrón.")
    if not CROP_AVAILABLE:
        st.warning("Instala `streamlit-cropper` para recortar. Mientras tanto se usará la imagen completa (sin recorte).")

    cat_col1, cat_col2 = st.columns([2,1])
    with cat_col1:
        catalogo_file = st.file_uploader("Sube imagen de catálogo (opcional, reusable para varias unidades)", type=["png","jpg","jpeg"], key="catalogo")
    with cat_col2:
        st.info("Tip: fondo blanco facilita la transparencia.")

    to_delete = []
    for i, simb in enumerate(st.session_state.leyenda_custom):
        st.markdown(f"**Unidad {i+1}:**")
        cols = st.columns([2, 1.1, 1, 1.2, 0.5])
        with cols[0]:
            simb['unidad'] = st.text_input("Nombre", value=simb['unidad'], key=f"unidadname_{i}")
        with cols[1]:
            simb['color'] = st.color_picker("Color del símbolo", value=simb['color'], key=f"colpick_{i}")
        with cols[2]:
            simb['hatch'] = st.text_input("Hatch (si no usas imagen)", value=simb['hatch'], key=f"hatch_{i}")
        with cols[4]:
            if st.button("🗑️", key=f"del_unidad_{i}"):
                to_delete.append(i)

        with cols[3]:
            st.write("**Símbolo por imagen**")
            use_catalog = st.checkbox("Usar catálogo subido", value=(catalogo_file is not None), key=f"usecat_{i}")
            file = catalogo_file if (use_catalog and catalogo_file is not None) else st.file_uploader("o sube imagen propia", type=["png","jpg","jpeg"], key=f"file_{i}")

            bgcol = st.color_picker("Color fondo a quitar", "#FFFFFF", key=f"bg_{i}")
            tol = st.slider("Tolerancia", 0, 120, 30, key=f"tol_{i}")
            scale = st.slider("Escala patrón (zoom)", 0.2, 3.0, simb.get('img_scale', 1.0), 0.1, key=f"scale_{i}")

            mode_options = {
                "Cover (recorta)": "cover",
                "Fit (estira)": "fit",
                "Contain (mantiene aspecto)": "contain",
                "Tile (repetir)": "tile"
            }
            inv_mode = {v: k for k, v in mode_options.items()}
            sel_key = inv_mode.get(simb.get('img_mode', 'cover'), "Cover (recorta)")
            mode_label = st.selectbox("Modo de ajuste", list(mode_options.keys()),
                                      index=list(mode_options.keys()).index(sel_key), key=f"mode_{i}")
            simb['img_mode'] = mode_options[mode_label]

            cropped = None
            if file is not None:
                raw = Image.open(file).convert("RGB")
                if CROP_AVAILABLE:
                    st.caption("Ajusta el rectángulo verde para elegir el motivo.")
                    cropped = st_cropper(raw, aspect_ratio=None, box_color="#00FF00", return_type="image", key=f"crop_{i}")
                else:
                    cropped = raw

                colb1, colb2 = st.columns(2)
                with colb1:
                    if st.button("Aplicar símbolo", key=f"apply_{i}"):
                        sym_rgb = hex_to_rgb(simb['color'])
                        bg_rgb = hex_to_rgb(bgcol)
                        processed = process_symbol_pil_only(cropped, bg_rgb=bg_rgb, tol=int(tol), symbol_rgb=sym_rgb)
                        bio = io.BytesIO()
                        processed.save(bio, format="PNG")
                        bio.seek(0)
                        simb['img_bytes'] = bio.read()
                        simb['img_scale'] = float(scale)
                        simb['img_native_w'], simb['img_native_h'] = processed.size
                        st.success("Símbolo aplicado.")
                with colb2:
                    if st.button("Quitar símbolo", key=f"removeimg_{i}"):
                        simb['img_bytes'] = None
                        simb['img_scale'] = 1.0
                        simb['img_mode'] = 'cover'
                        simb['img_native_w'] = None
                        simb['img_native_h'] = None
                        st.info("Símbolo eliminado.")
            st.markdown("---")

    # Evitar borrar unidades en uso
    used_units = set()
    if "df" in st.session_state:
        used_units = set(st.session_state.df.get("Unidad", []))

    for idx in sorted(to_delete, reverse=True):
        name = st.session_state.leyenda_custom[idx]['unidad']
        if name in used_units:
            st.warning(f"No se puede eliminar '{name}' porque está en uso en la tabla.")
        else:
            del st.session_state.leyenda_custom[idx]

    if st.button("➕ Agregar nueva unidad"):
        st.session_state.leyenda_custom.append(
            {'unidad': f"Nueva Unidad {len(st.session_state.leyenda_custom)+1}", 'uh': '', 'color': '#000000', 'hatch': '',
             'img_bytes': None, 'img_scale': 1.0, 'img_mode': 'cover', 'img_native_w': None, 'img_native_h': None}
        )

    ccol1, ccol2 = st.columns([1,1])
    with ccol1:
        if st.button("🔄 Restaurar simbología por defecto"):
            st.session_state.leyenda_custom = [
                {**d, 'img_bytes': None, 'img_scale': 1.0, 'img_mode': 'cover', 'img_native_w': None, 'img_native_h': None}
                for d in leyenda_default
            ]
    with ccol2:
        st.info("Hatch típicos: `/`, `\\`, `x`, `xx`, `|`, `-`, `+`, `.`, `...`, `//`, `|||`, etc.")

# -------------------------
# Usar simbología actual
# -------------------------
leyenda_actual = st.session_state.leyenda_custom
unidades_lista = [d['unidad'] for d in leyenda_actual]
leyenda_lookup = {d['unidad']: d for d in leyenda_actual}

# -------------------------
# Data de ejemplo
# -------------------------
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

# Sincroniza nombres si ya hay df
old_names = [d['unidad'] for d in leyenda_default]
new_names = [d['unidad'] for d in leyenda_actual[:len(leyenda_default)]]
mapping = {old: new for old, new in zip(old_names, new_names)}
st.session_state.df["Unidad"] = st.session_state.df["Unidad"].replace(mapping)
st.session_state.df.loc[~st.session_state.df["Unidad"].isin(unidades_lista), "Unidad"] = ""

# -------- Leyenda de vista previa ----------
st.markdown("### Leyenda de patrones:")
prev_w = max(10, 0.7*len(leyenda_actual) + 3)
fig_leyenda, axl = plt.subplots(figsize=(prev_w, 2.1))
axl.axis('off')
axl.set_aspect('auto')
for i, simb in enumerate(leyenda_actual):
    axl.add_patch(
        mpatches.Rectangle((i, 0.18), 0.98, 0.56, facecolor="#ffffff",
                           edgecolor='black', linewidth=1.2, zorder=0.9)
    )
    if simb.get('img_bytes'):
        img = Image.open(io.BytesIO(simb['img_bytes'])).convert("RGBA")
        _ = fill_rect_with_image(axl, img, i+0.01, 0.20, 0.96, 0.52,
                                 scale=simb.get('img_scale', 1.0), zorder=1.0,
                                 mode=simb.get('img_mode', 'cover'), align_phase=False)
        label_pat = "img"
    else:
        axl.add_patch(
            mpatches.Rectangle((i+0.01, 0.20), 0.96, 0.52, facecolor=simb['color'],
                               hatch=simb['hatch'], edgecolor='black', linewidth=1.0, zorder=1.0)
        )
        label_pat = f"`{simb['hatch']}`"
    axl.text(i+0.50, 0.78, label_pat, ha='center', va='bottom',
             fontsize=5, color='dimgray', family='monospace')
    axl.text(i+0.50, 0.12, simb['unidad'], ha='center', va='top', fontsize=4.2, rotation=90)
axl.set_xlim(0, len(leyenda_actual))
axl.set_ylim(0, 1)
st.pyplot(fig_leyenda, use_container_width=True, dpi=200)

# -------------------------
# GESTOR RÁPIDO DE UNIDADES
# -------------------------
with st.expander("Gestión rápida de Unidades (agregar/quitar desde el editor de filas)", expanded=True):
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        new_u = st.text_input("Nueva unidad (solo nombre)", key="quick_new_u")
    with c2:
        if st.button("➕ Agregar unidad", key="quick_add_u"):
            if new_u and new_u not in unidades_lista:
                st.session_state.leyenda_custom.append(
                    {'unidad': new_u, 'uh': '', 'color': '#cccccc', 'hatch': '',
                     'img_bytes': None, 'img_scale': 1.0, 'img_mode': 'cover',
                     'img_native_w': None, 'img_native_h': None}
                )
                st.success(f"Unidad '{new_u}' agregada.")
            else:
                st.warning("Escribe un nombre nuevo que no exista.")
    with c3:
        del_u = st.selectbox("Eliminar unidad (si no está en uso)", [""] + unidades_lista, key="quick_del_sel")
        if st.button("🗑️ Eliminar unidad", key="quick_del_u"):
            if del_u:
                if del_u in set(st.session_state.df["Unidad"]):
                    st.warning(f"No se puede eliminar '{del_u}' porque está en uso en la tabla.")
                else:
                    st.session_state.leyenda_custom = [d for d in st.session_state.leyenda_custom if d['unidad'] != del_u]
                    st.success(f"Unidad '{del_u}' eliminada.")

# Recalcular listas tras posibles cambios
leyenda_actual = st.session_state.leyenda_custom
unidades_lista = [d['unidad'] for d in leyenda_actual]
leyenda_lookup = {d['unidad']: d for d in leyenda_actual}

# -------------------------
# Editor de filas (con agregar/quitar)
# -------------------------
st.write("**Edita los datos:**")

def _make_default_row(last_inf: float = None, alto: float = 20.0):
    if last_inf is None:
        sup = 0.0
    else:
        sup = float(last_inf)
    inf = sup + float(alto)
    return {
        "Profundidad_sup": sup,
        "Profundidad_inf": inf,
        "Litologia": "",
        "Unidad": "",
        "UH": ""
    }

# Botonera superior del editor
ec1, ec2, ec3, ec4 = st.columns([1,1,1,1.5])
with ec1:
    if st.button("➕ Agregar fila al final"):
        last_inf = float(st.session_state.df["Profundidad_inf"].max()) if len(st.session_state.df) else None
        newrow = _make_default_row(last_inf)
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([newrow])], ignore_index=True)
with ec2:
    if st.button("↕️ Ordenar por profundidad"):
        st.session_state.df = st.session_state.df.sort_values(by=["Profundidad_sup","Profundidad_inf"], ascending=[True, True]).reset_index(drop=True)
with ec3:
    if st.button("🧮 Normalizar continuidad"):
        # ajusta para que cada sup sea el inf anterior (no reduce alturas negativas)
        df = st.session_state.df.sort_values(by=["Profundidad_sup","Profundidad_inf"]).reset_index(drop=True)
        for i in range(1, len(df)):
            df.at[i, "Profundidad_sup"] = df.at[i-1, "Profundidad_inf"]
            if df.at[i, "Profundidad_inf"] <= df.at[i, "Profundidad_sup"]:
                df.at[i, "Profundidad_inf"] = df.at[i, "Profundidad_sup"] + 1.0
        st.session_state.df = df.copy()
with ec4:
    st.caption("Consejo: inserta filas con el botón ➕ debajo de cada registro. Usa **Normalizar** para encadenar profundidades.")

df_input = st.session_state.df.copy()
to_delete_rows = []
to_insert_below = []

for idx in df_input.index:
    cols = st.columns([1, 1.1, 2.0, 1.2, 0.65, 0.65])
    with cols[0]:
        df_input.at[idx, "Profundidad_sup"] = st.number_input(
            f"Sup {idx+1}", value=float(df_input.at[idx, "Profundidad_sup"]), key=f"sup_{idx}", step=1.0
        )
    with cols[1]:
        df_input.at[idx, "Profundidad_inf"] = st.number_input(
            f"Inf {idx+1}", value=float(df_input.at[idx, "Profundidad_inf"]), key=f"inf_{idx}", step=1.0
        )
    with cols[2]:
        df_input.at[idx, "Litologia"] = st.text_input(
            f"Litología {idx+1}", value=df_input.at[idx, "Litologia"], key=f"lito_{idx}"
        )
    with cols[3]:
        current = df_input.at[idx, "Unidad"]
        if current not in unidades_lista:
            current = ""
        df_input.at[idx, "Unidad"] = st.selectbox(
            f"Unidad {idx+1}", options=[""] + unidades_lista,
            index=([""] + unidades_lista).index(current), key=f"unidad_{idx}"
        )
    with cols[4]:
        if st.button("➕ debajo", key=f"add_below_{idx}"):
            to_insert_below.append(idx)
    with cols[5]:
        if st.button("🗑️ fila", key=f"del_row_{idx}"):
            to_delete_rows.append(idx)

    # UH en línea completa para mejor lectura
    df_input.at[idx, "UH"] = st.text_input(f"UH {idx+1}", value=df_input.at[idx, "UH"], key=f"uh_{idx}")
    st.markdown("---")

# Aplicar inserciones
if to_insert_below:
    df_work = df_input.copy()
    # procesar en orden descendente para no desplazar índices por delante
    for i in sorted(to_insert_below, reverse=True):
        sup_i = float(df_work.at[i, "Profundidad_sup"])
        inf_i = float(df_work.at[i, "Profundidad_inf"])
        alto = max(1.0, inf_i - sup_i)
        newrow = _make_default_row(inf_i, alto)
        upper = df_work.iloc[:i+1]
        lower = df_work.iloc[i+1:]
        df_work = pd.concat([upper, pd.DataFrame([newrow]), lower], ignore_index=True)
    df_input = df_work.copy()

# Aplicar eliminaciones
if to_delete_rows:
    df_input = df_input.drop(index=to_delete_rows).reset_index(drop=True)

# Guardar cambios
st.session_state.df = df_input

# columnas auxiliares para plot
df_plot = st.session_state.df.copy()
df_plot["Color"]       = df_plot["Unidad"].map(lambda x: leyenda_lookup[x]["color"] if x in leyenda_lookup else "#ffffff")
df_plot["Hatch"]       = df_plot["Unidad"].map(lambda x: leyenda_lookup[x]["hatch"] if x in leyenda_lookup else "")
df_plot["ImgBytes"]    = df_plot["Unidad"].map(lambda x: leyenda_lookup[x].get("img_bytes") if x in leyenda_lookup else None)
df_plot["ImgScale"]    = df_plot["Unidad"].map(lambda x: leyenda_lookup[x].get("img_scale", 1.0) if x in leyenda_lookup else 1.0)
df_plot["ImgMode"]     = df_plot["Unidad"].map(lambda x: leyenda_lookup[x].get("img_mode", "cover") if x in leyenda_lookup else "cover")
df_plot["ImgNativeW"]  = df_plot["Unidad"].map(lambda x: leyenda_lookup[x].get("img_native_w") if x in leyenda_lookup else None)
df_plot["ImgNativeH"]  = df_plot["Unidad"].map(lambda x: leyenda_lookup[x].get("img_native_h") if x in leyenda_lookup else None)

# -------------------------
# Plot principal
# -------------------------
prof_max = float(df_plot["Profundidad_inf"].max())
prof_min = float(df_plot["Profundidad_sup"].min())

fig = plt.figure(figsize=(fig_width, 10), dpi=150)
gs = fig.add_gridspec(
    2, 5,
    height_ratios=[14, 1.7],
    #         [Descripción, gap, Regla,       Unidad,            UH   ]
    width_ratios=[  2.4,     0.25, 0.7, col_unidad_ratio, col_uh_ratio ],
    hspace=0.08,
    wspace=0.20
)
ax_lit  = fig.add_subplot(gs[0, 0])
ax0     = fig.add_subplot(gs[0, 2], sharey=ax_lit)
ax1     = fig.add_subplot(gs[0, 3], sharey=ax_lit)
ax_text = fig.add_subplot(gs[0, 4], sharey=ax_lit)
ax_leg  = fig.add_subplot(gs[1, :])

# Descripción litológica
ax_lit.set_ylim(prof_max, prof_min)
ax_lit.set_xlim(0, 1)
ax_lit.axis('off')
ax_lit.set_title("Descripción Litología", fontsize=10, weight="bold", pad=16)
for i, row in df_plot.iterrows():
    height = row["Profundidad_inf"] - row["Profundidad_sup"]
    fontsize = max(8, min(13, height * 0.17))
    wrapper = textwrap.TextWrapper(width=40)
    text_wrapped = "\n".join(wrapper.wrap(str(row["Litologia"])))
    max_lines = int(height // 6.5)
    lines = text_wrapped.split('\n')
    if len(lines) > max_lines:
        text_wrapped = '\n'.join(lines[:max_lines]) + '\n...'
    ax_lit.add_patch(
        mpatches.Rectangle((0.04, row["Profundidad_sup"]), 0.92, height,
                           fill=False, edgecolor='black', linewidth=1.1, zorder=1)
    )
    ax_lit.text(
        0.5, (row["Profundidad_sup"] + row["Profundidad_inf"]) / 2,
        text_wrapped, va="center", ha="center",
        fontsize=fontsize, wrap=True, zorder=2, clip_on=True,
        bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", linewidth=0)
    )

# Regla de profundidad
ax0.set_ylim(prof_max, prof_min); ax0.set_xlim(0, 1)
ax0.set_yticks([]); ax0.set_xticks([]); ax0.axis('off')
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

# Columna de unidades
ax1.set_ylim(prof_max, prof_min); ax1.set_xlim(0, 1)
ax1.set_aspect('auto'); ax1.axis("off")
ax1.set_title("Unidad Hidrogeológica", fontsize=10, weight="bold", pad=16)

# --- Cálculo de ancho global cuando hay Contain ---
p0_col = ax1.transData.transform((0.0, prof_min))
p1_col = ax1.transData.transform((1.0, prof_min))
col_px_w = max(1, int(abs(p1_col[0] - p0_col[0])))

global_contain_frac = None
if contain_match_width and contain_apply_global and (df_plot["ImgMode"] == "contain").any():
    fracs = []
    for _, r in df_plot.iterrows():
        if r["ImgMode"] == "contain" and r["ImgNativeW"] is not None:
            fracs.append((float(r["ImgNativeW"]) / float(col_px_w)) * float(r["ImgScale"]))
    if fracs:
        global_contain_frac = float(np.median(fracs))  # robusto ante outliers
    else:
        global_contain_frac = contain_fallback_pct / 100.0
    global_contain_frac = float(np.clip(global_contain_frac, 0.05, 1.0))

# --- Dibujo de cada estrato ---
for i, row in df_plot.iterrows():
    y0 = row["Profundidad_sup"]
    h  = row["Profundidad_inf"] - row["Profundidad_sup"]

    # ¿Usamos ancho override?
    width_override = None
    if contain_match_width and contain_apply_global and global_contain_frac is not None:
        width_override = global_contain_frac
    elif contain_match_width and row["ImgMode"] == "contain" and row["ImgNativeW"] is not None:
        # modo por unidad (cuando el global está desactivado)
        width_override = (float(row["ImgNativeW"]) / float(col_px_w)) * float(row["ImgScale"])
        width_override = float(np.clip(width_override, 0.05, 1.0))

    if row["ImgBytes"] is not None:
        img = Image.open(io.BytesIO(row["ImgBytes"])).convert("RGBA")
        info = fill_rect_with_image(
            ax1, img, 0.0, y0, 1.0, h,
            scale=float(row["ImgScale"]),
            zorder=1.0,
            mode=row["ImgMode"],
            align_phase=True,
            align=_align_map[contain_align_label],
            width_override=width_override
        )
        # Borde
        ax1.add_patch(
            mpatches.Rectangle((info["rect_x0"], y0), info["rect_width"], h,
                               facecolor="none", edgecolor="black", linewidth=1.2, zorder=1.5)
        )
    else:
        # Hatch/Color: aplicar mismo ancho si corresponde
        if width_override is not None:
            # alinear según opción
            if _align_map[contain_align_label] == "left":
                x_left = 0.0
            elif _align_map[contain_align_label] == "right":
                x_left = 1.0 - width_override
            else:
                x_left = (1.0 - width_override) / 2.0
            ax1.add_patch(
                mpatches.Rectangle((x_left, y0), width_override, h, facecolor=row["Color"],
                                   edgecolor="black", linewidth=1.2, hatch=row["Hatch"], zorder=1.0)
            )
        else:
            ax1.add_patch(
                mpatches.Rectangle((0.0, y0), 1.0, h, facecolor=row["Color"],
                                   edgecolor="black", linewidth=1.2, hatch=row["Hatch"], zorder=1.0)
            )

# Texto UH
ax_text.set_ylim(prof_max, prof_min); ax_text.set_xlim(0, 1); ax_text.axis("off")
for i, row in df_plot.iterrows():
    ax_text.text(0.01, (row["Profundidad_sup"] + row["Profundidad_inf"]) / 2,
                 row["UH"], va="center", ha="left", fontsize=9, fontweight="bold", color="black")

# Leyenda inferior
ax_leg.axis('off'); ax_leg.set_xlim(0, len(leyenda_actual)); ax_leg.set_ylim(0, 1.6); ax_leg.set_aspect('auto')
for i, simb in enumerate(leyenda_actual):
    ax_leg.add_patch(
        mpatches.Rectangle((i+0.02, 0.82), 0.75, 0.45, facecolor="white",
                           edgecolor='black', linewidth=1.1, zorder=0.9)
    )
    if simb.get('img_bytes'):
        img = Image.open(io.BytesIO(simb['img_bytes'])).convert("RGBA")
        _ = fill_rect_with_image(
            ax_leg, img, i+0.02, 0.82, 0.75, 0.45,
            scale=simb.get('img_scale', 1.0),
            zorder=1.0,
            mode=simb.get('img_mode', 'cover'),
            align_phase=False
        )
    else:
        ax_leg.add_patch(
            mpatches.Rectangle((i+0.02, 0.82), 0.75, 0.45, facecolor=simb['color'],
                               hatch=simb['hatch'], edgecolor='black', linewidth=1.1, zorder=1.0)
        )
    ax_leg.text(i+0.38, 0.78, simb['unidad'], ha='center', va='top', fontsize=6.5, rotation=90, weight='bold')
ax_leg.set_title("Leyenda", fontsize=10, loc='left', pad=7, rotation=0, weight='bold')

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.05, right=0.98, wspace=0.20, hspace=0.08)
st.pyplot(fig, use_container_width=True, dpi=200)

# ==== DESCARGAS PNG y SVG ====
png_buffer = io.BytesIO()
fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight"); png_buffer.seek(0)
svg_buffer = io.BytesIO()
fig.savefig(svg_buffer, format="svg", bbox_inches="tight"); svg_buffer.seek(0)

st.markdown("### Descargar figura:")
colp, cols = st.columns(2)
with colp:
    st.download_button("📥 PNG", data=png_buffer, file_name="columna_estratigrafica.png", mime="image/png")
with cols:
    st.download_button("📥 SVG", data=svg_buffer, file_name="columna_estratigrafica.svg", mime="image/svg+xml")
