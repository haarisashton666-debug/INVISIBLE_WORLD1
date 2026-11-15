import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ==========================================
# INVISIBLE WORLD 2.0
# AI SCIENTIFIC IMAGE ANALYSIS
# ==========================================

st.set_page_config(
    page_title="INVISIBLE WORLD",
    page_icon="🌌",
    layout="wide"
)

# ==========================================
# DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07152e, #102e65);
    color: white;
}
h1, h2, h3 {
    color: #19dcff !important;
}
div[data-testid="stMetric"] {
    background: #123d79;
    border: 1px solid #12cfff;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌌 INVISIBLE WORLD")
st.caption("AI VISION SCANNER | SCIENTIFIC IMAGE ANALYSIS")
st.markdown("---")

# ==========================================
# LOAD PRETRAINED AI MODELS
# ==========================================

@st.cache_resource
def load_models():

    detector = YOLO("yolo11n.pt")

    classifier = YOLO("yolo11n-cls.pt")

    return detector, classifier


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🧭 NAVIGATION")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Photo Scanner",
        "About Project"
    ]
)

st.sidebar.info(
    "Explore the hidden details of ordinary "
    "photographs using AI and computer vision."
)

# ==========================================
# HOME
# ==========================================

if page == "Home":

    st.header("Welcome to Invisible World")

    st.write(
        "Discover the hidden characteristics of "
        "ordinary objects using artificial intelligence "
        "and scientific image processing."
    )

    col1, col2, col3 = st.columns(3)

    col1.info("📷 CAMERA SCANNER\n\nCapture photographs.")

    col2.info("🧠 AI INTELLIGENCE\n\nIdentify objects.")

    col3.info("📊 DATA ANALYTICS\n\nGenerate scientific measurements.")

    st.markdown("---")

    st.subheader("Available Technologies")

    st.write("""
    - AI Object Detection
    - AI Image Classification
    - RGB Colour Analysis
    - Grayscale Analysis
    - Image Intensity Histogram
    - Hidden Pattern Detection
    - Edge and Contour Detection
    - Texture Analysis
    - Scientific Measurements
    """)

    st.success(
        "Open Photo Scanner to capture or upload an image."
    )


# ==========================================
# PHOTO SCANNER
# ==========================================

elif page == "Photo Scanner":

    st.header("📷 Advanced Photo Scanner")

    st.write(
        "Capture an object using your camera "
        "or upload an existing photograph."
    )

    camera_image = st.camera_input("Take Photo")

    uploaded_image = st.file_uploader(
        "Or upload an image",
        type=["jpg", "jpeg", "png"]
    )

    image_source = camera_image if camera_image is not None else uploaded_image

    if image_source is not None:

        image = Image.open(image_source).convert("RGB")

        rgb = np.array(image)

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        height, width = gray.shape

        total_pixels = width * height

        st.markdown("---")

        st.subheader("📸 Original Photograph")

        st.image(
            image,
            use_container_width=True
        )

        # ==================================
        # EXISTING SCIENTIFIC ANALYTICS
        # ==================================

        st.markdown("---")
        st.header("📊 SCIENTIFIC ANALYSIS")

        brightness = float(np.mean(gray))

        contrast = float(np.std(gray))

        edge_map = cv2.Canny(
            gray,
            50,
            150
        )

        edge_percentage = (
            np.count_nonzero(edge_map) /
            total_pixels
        ) * 100

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Width", f"{width} px")

        c2.metric("Height", f"{height} px")

        c3.metric("Brightness", f"{brightness:.2f}/255")

        c4.metric("Contrast", f"{contrast:.2f}")

        c5, c6, c7 = st.columns(3)

        c5.metric("Edge Percentage", f"{edge_percentage:.2f}%")

        c6.metric("Total Pixels", f"{total_pixels:,}")

        c7.metric("Image Channels", "3")

        # ==================================
        # RGB COLOUR ANALYSIS
        # ==================================

        st.markdown("---")
        st.header("🎨 RGB COLOUR ANALYSIS")

        rgb_values = {
            "Red": float(np.mean(rgb[:, :, 0])),
            "Green": float(np.mean(rgb[:, :, 1])),
            "Blue": float(np.mean(rgb[:, :, 2]))
        }

        rgb_df = pd.DataFrame({
            "Channel": list(rgb_values.keys()),
            "Average Intensity": list(rgb_values.values())
        })

        st.bar_chart(
            rgb_df.set_index("Channel")
        )

        for channel, value in rgb_values.items():

            st.write(
                f"**{channel}:** {value:.2f}/255"
            )

        # ==================================
        # GRAYSCALE ANALYSIS
        # ==================================

        st.markdown("---")
        st.header("🔬 GRAYSCALE ANALYSIS")

        st.image(
            gray,
            caption="Grayscale Processed Image",
            use_container_width=True
        )

        # ==================================
        # IMAGE INTENSITY HISTOGRAM
        # ==================================

        st.markdown("---")
        st.header("📈 IMAGE INTENSITY HISTOGRAM")

        histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        ).flatten()

        histogram_df = pd.DataFrame({
            "Intensity": np.arange(256),
            "Number of Pixels": histogram
        })

        st.line_chart(
            histogram_df.set_index("Intensity")
        )

        # ==================================
        # AI OBJECT INTELLIGENCE
        # ==================================

        st.markdown("---")
        st.header("🧠 AI OBJECT INTELLIGENCE")

        st.write(
            "Identify objects and predict their categories "
            "using pretrained artificial intelligence."
        )

        if st.button("🔍 START AI OBJECT IDENTIFICATION"):

            try:

                with st.spinner(
                    "Loading AI models and analysing photograph..."
                ):

                    detector, classifier = load_models()

                    # Object detection
                    detection_results = detector.predict(
                        source=image,
                        conf=0.25,
                        verbose=False
                    )

                    # Image classification
                    classification_results = classifier.predict(
                        source=image,
                        verbose=False
                    )

                    detection = detection_results[0]

                    classification = classification_results[0]

                    # Draw detected objects
                    detected_image = image.copy()

                    draw = ImageDraw.Draw(detected_image)

                    detected_objects = []

                    if detection.boxes is not None:

                        for box in detection.boxes:

                            coordinates = box.xyxy[0].tolist()

                            confidence = float(box.conf[0])

                            class_id = int(box.cls[0])

                            object_name = detection.names[class_id]

                            detected_objects.append({
                                "Object": object_name,
                                "Confidence": confidence
                            })

                            draw.rectangle(
                                coordinates,
                                outline="lime",
                                width=4
                            )

                            draw.text(
                                (coordinates[0], max(0, coordinates[1] - 20)),
                                f"{object_name} {confidence:.1%}",
                                fill="lime"
                            )

                    st.success(
                        "AI Object Analysis Completed!"
                    )

                    st.subheader("🎯 Object Detection Result")

                    st.image(
                        detected_image,
                        caption="AI Detected Objects",
                        use_container_width=True
                    )

                    st.subheader("📋 Identified Objects")

                    if detected_objects:

                        detection_df = pd.DataFrame(
                            detected_objects
                        )

                        detection_df["Confidence"] = (
                            detection_df["Confidence"] * 100
                        ).round(2).astype(str) + "%"

                        st.dataframe(
                            detection_df,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "No supported objects were detected. "
                            "Try another photograph with a clearer view."
                        )

                    # Classification predictions
                    st.markdown("---")

                    st.subheader("🏷️ AI Image Classification")

                    if classification.probs is not None:

                        top_indices = classification.probs.top5

                        top_confidences = (
                            classification.probs.top5conf
                        )

                        for index, score in zip(
                            top_indices,
                            top_confidences
                        ):

                            class_name = classification.names[index]

                            confidence = float(score) * 100

                            st.write(
                                f"**{class_name}** — "
                                f"{confidence:.2f}% confidence"
                            )

                            st.progress(
                                min(int(confidence), 100)
                            )

                    st.info(
                        "Classification scores are model predictions, "
                        "not guarantees of correct identification."
                    )

                    # Visible characteristics
                    st.markdown("---")

                    st.subheader(
                        "🔎 Visible Image Characteristics"
                    )

                    if detected_objects:

                        st.write(
                            "Detected object count:",
                            len(detected_objects)
                        )

                        for item in detected_objects:

                            st.write(
                                f"- {item['Object']}: "
                                f"detected by the AI model."
                            )

                    st.write(
                        f"Image dimensions: {width} × {height} pixels"
                    )

                    st.write(
                        f"Average brightness: {brightness:.2f}/255"
                    )

                    st.write(
                        f"Average contrast: {contrast:.2f}"
                    )

                    st.write(
                        "RGB channel averages:",
                        rgb_values
                    )

            except Exception as e:

                st.error(
                    f"AI identification error: {e}"
                )

                st.info(
                    "Check your internet connection and try again. "
                    "The first model download may take some time."
                )

        # ==================================
        # HIDDEN PATTERN DETECTION
        # ==================================

        st.markdown("---")
        st.header("✨ HIDDEN PATTERN DETECTION")

        st.write(
            "Explore fine image structures, boundaries, "
            "contours and surface patterns."
        )

        if st.button("🔬 REVEAL HIDDEN PATTERNS"):

            # Contrast enhancement
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced = clahe.apply(gray)

            # Edge detection
            edges = cv2.Canny(
                enhanced,
                50,
                150
            )

            # Contour detection
            contours, hierarchy = cv2.findContours(
                edges.copy(),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )

            contour_image = rgb.copy()

            cv2.drawContours(
                contour_image,
                contours,
                -1,
                (0, 255, 0),
                2
            )

            # Adaptive threshold
            threshold = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            # Texture enhancement
            kernel = np.ones(
                (5, 5),
                np.uint8
            )

            texture = cv2.morphologyEx(
                enhanced,
                cv2.MORPH_TOPHAT,
                kernel
            )

            # Display results
            st.success(
                "Hidden Pattern Analysis Completed!"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    edges,
                    caption="Edge Detection",
                    use_container_width=True
                )

            with col2:

                st.image(
                    contour_image,
                    caption="Contour Detection",
                    use_container_width=True
                )

            col3, col4 = st.columns(2)

            with col3:

                st.image(
                    threshold,
                    caption="Adaptive Threshold",
                    use_container_width=True
                )

            with col4:

                st.image(
                    texture,
                    caption="Texture Enhancement",
                    use_container_width=True
                )

            # Pattern measurements
            st.markdown("---")

            st.subheader("📊 PATTERN ANALYTICS")

            edge_percentage_new = (
                np.count_nonzero(edges) /
                total_pixels
            ) * 100

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Detected Contours",
                len(contours)
            )

            p2.metric(
                "Edge Percentage",
                f"{edge_percentage_new:.2f}%"
            )

            p3.metric(
                "Total Pixels",
                f"{total_pixels:,}"
            )

            st.info(
                "These are processed visual patterns, "
                "not proof of physically invisible objects."
            )

        # ==================================
        # DOWNLOAD ORIGINAL IMAGE
        # ==================================

        st.markdown("---")

        buffer = __import__("io").BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            "⬇️ DOWNLOAD ORIGINAL IMAGE",
            data=buffer.getvalue(),
            file_name="invisible_world_image.png",
            mime="image/png"
        )

    else:

        st.info(
            "Take a photograph or upload an image to begin analysis."
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header("🌌 About Invisible World")

    st.write("""
    INVISIBLE WORLD is an AI-powered scientific
    image analysis application.

    It combines computer vision, machine learning
    and digital image processing to explore the
    characteristics of ordinary photographs.

    Main technologies:

    1. YOLO Object Detection
    2. YOLO Image Classification
    3. OpenCV Image Processing
    4. RGB Colour Analysis
    5. Grayscale Analysis
    6. Histogram Analysis
    7. Edge and Contour Detection
    8. Texture Enhancement
    """)

    st.success(
        "Explore the world beyond ordinary photographs!"
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "INVISIBLE WORLD | AI VISION & SCIENTIFIC ANALYSIS"
)
