import streamlit as st
import numpy as np
import cv2
import pandas as pd
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fingerprint Blood Group Detection",
    page_icon="🩸",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#F8FBFF,
#EEF4FF,
#F5F7FA
);
color:#222222;
}

.hero{

padding:30px;

border-radius:25px;

background:linear-gradient(
135deg,
#5B86E5,
#36D1DC
);

color:white;

box-shadow:0px 10px 25px rgba(0,0,0,.15);

animation:fadein 1.5s;
}

.card{

background:white;

padding:25px;

border-radius:20px;

box-shadow:0px 5px 20px rgba(0,0,0,.08);

margin-top:10px;

transition:.3s;
}

.card:hover{

transform:translateY(-6px);

}

@keyframes fadein{

from{
opacity:0;
transform:translateY(25px);
}

to{
opacity:1;
transform:translateY(0px);
}

}

.big{

font-size:45px;
font-weight:bold;

}

.small{

font-size:20px;

}

section[data-testid="stSidebar"]{

background:#ffffff;

border-right:1px solid #E5E7EB;

}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

@st.cache_resource
def load_trained_model():

    model=load_model(
        "model/blood_model.h5",
        compile=False
    )

    return model

model=load_trained_model()

classes=[

"A+",
"A-",
"AB+",
"AB-",
"B+",
"B-",
"O+",
"O-"

]

# ---------------- SIDEBAR ----------------

st.sidebar.image(
"https://cdn-icons-png.flaticon.com/512/3774/3774299.png",
width=90
)

st.sidebar.title("🧬 AI Navigation")

menu=st.sidebar.radio(
"",
[
"🏠 Home",
"🔍 Prediction",
"📘 About"
]
)

st.sidebar.markdown("---")

st.sidebar.success("""

Model: MobileNetV2

Dataset: 8000+

Classes: 8

Accuracy: ~85%

""")

# ---------------- HOME ----------------

if menu=="🏠 Home":

    st.markdown("""

<div class='hero'>

<div class='big'>
🩸 Fingerprint Blood Group Detection
</div>

<div class='small'>
Fingerprint Based Blood Group Prediction using Deep Learning
</div>

</div>

""",unsafe_allow_html=True)

    st.write("")

    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.metric(
        "Classes",
        "8"
        )

    with c2:
        st.metric(
        "Model",
        "MobileNetV2"
        )

    with c3:
        st.metric(
        "Dataset",
        "8000+"
        )

    with c4:
        st.metric(
        "Accuracy",
        "85%"
        )

    st.success("""

✔ Upload fingerprint image

✔ Automatic preprocessing

✔ Feature extraction

✔ Blood group prediction

✔ Real-time deployment

""")

# ---------------- PREDICTION ----------------

elif menu=="🔍 Prediction":

    st.markdown("""

<div class='hero'>

<h2>📤 Upload Fingerprint Image</h2>

Upload image and predict blood group

</div>

""", unsafe_allow_html=True)

    uploaded_file=st.file_uploader(
    "",
    type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        with st.spinner(
        "Analyzing fingerprint..."
        ):

            time.sleep(1)

            file_bytes=np.asarray(
            bytearray(
            uploaded_file.read()
            ),
            dtype=np.uint8
            )

            img=cv2.imdecode(
            file_bytes,
            1
            )

            col1,col2=st.columns(2)

            with col1:

                st.markdown(
                "<div class='card'>",
                unsafe_allow_html=True
                )

                st.image(
                img,
                width=350
                )

                st.markdown(
                "</div>",
                unsafe_allow_html=True
                )

            h,w,_=img.shape

            img=img[
            int(h*.1):int(h*.9),
            int(w*.1):int(w*.9)
            ]

            img=cv2.resize(
            img,
            (224,224)
            )

            if len(img.shape)==2:

                img=cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2RGB
                )

            img=preprocess_input(
            img
            )

            img=np.expand_dims(
            img,
            axis=0
            )

            prediction=model.predict(img)

            result_index=np.argmax(
            prediction
            )

            result=classes[
            result_index
            ]

            confidence=float(
            np.max(prediction)
            )*100

            with col2:

                st.markdown(
                "<div class='card'>",
                unsafe_allow_html=True
                )

                st.subheader(
                "🧠 AI Prediction"
                )

                st.success(
                f"Predicted Blood Group: {result}"
                )

                st.progress(
                int(confidence)
                )

                st.write(
                f"Confidence: {confidence:.2f}%"
                )

                st.metric(
                "Inference Time",
                "<1 sec"
                )

                st.markdown(
                "</div>",
                unsafe_allow_html=True
                )

            st.markdown("<br>",unsafe_allow_html=True)

            st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True
            )

            st.subheader(
            "📊 Class Probabilities"
            )

            prob_df=pd.DataFrame({

            "Blood Group":
            classes,

            "Probability (%)":
            prediction[0]*100

            })

            st.bar_chart(
            prob_df.set_index(
            "Blood Group"
            )
            )

            st.markdown(
            "</div>",
            unsafe_allow_html=True
            )

# ---------------- ABOUT ----------------

else:

    st.markdown("""

<div class='hero'>
<h1>📘 About Project</h1>
</div>

""",unsafe_allow_html=True)

    st.write("""

### Workflow

Fingerprint Image

⬇

Preprocessing

⬇

MobileNetV2

⬇

Dense Layers

⬇

Softmax Classification

### Technologies

- TensorFlow
- OpenCV
- Streamlit
- Deep Learning
- MobileNetV2

""")

st.markdown("---")

st.caption(
"Final Year Project • AI + Biometrics"
)