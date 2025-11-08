import streamlit as st
import  pandas as pd
import numpy as np
import pickle
import base64
import time
#---------------------- APP TITLE ----------------------
st.set_page_config(page_title="EpiDemic - Disease Outbreak Prediction", page_icon="🦠", layout="wide")
st.title(" EpiDemic- Disease Outbreak Prediction")
tab1, tab2, tab3, tab4 = st.tabs(["About", "covid-19", "H1N1-FLU-VACCINES", "Dashboard"])

with tab1:
    import streamlit as st

    st.header("📖 About EpiDemic: Disease Outbreak Prediction App")
    
    st.write("Welcome! This app predicts COVID-19 outbreak risk and H1N1 flu vaccination likelihood with an **interactive, fun interface**! 🦠💉")
    
    # ---------------------- Fun Expanders ----------------------
    with st.expander("🎯 Purpose of this App"):
        st.write("""
        - Predict COVID-19 outbreak risk based on your symptoms, age, and contact history.  
        - Predict the likelihood of getting H1N1 flu vaccination based on behavioral and health factors.  
        - Make predictions using multiple machine learning models like Logistic Regression, Random Forest, and XGBoost.  
        """)
    
    with st.expander("🛠️ How It Works"):
        st.write("""
        1. Fill the interactive forms for COVID-19 or H1N1.  
        2. Submit to see predictions with fun **emoji feedback**.  
        3. The app uses pre-trained ML models to estimate risk or likelihood.  
        4. Progress bars simulate “analyzing your data” for a gamified experience.  
        """)
    
    with st.expander("💡 Fun Facts"):
        st.write("""
        - COVID-19 risk increases with age and comorbidities. 🔬  
        - H1N1 vaccine uptake depends on knowledge, concern, and doctor recommendations. 💉  
        - Preventive behaviors like mask usage and hand washing are critical! 😷🧼  
        """)
    
    with st.expander("❓ Did You Know? Quiz"):
        st.write("Answer correctly and see how much you know about viruses!")
        q1 = st.radio("COVID-19 spreads mainly through:", ["Airborne droplets", "Text messages", "WiFi signals"])
        q2 = st.radio("H1N1 vaccine helps prevent:", ["Flu symptoms", "Headache from computer", "Bad dreams"])
        if st.button("Check Answers"):
            score = 0
            if q1 == "Airborne droplets": score += 1
            if q2 == "Flu symptoms": score += 1
            st.success(f"🎉 You scored {score}/2!")
    
    st.markdown("---")
    st.write("👩‍💻 Developed as a fun, interactive tool for learning and health awareness.")
    
#---------------------- COVID-19 PREDICTION TAB ----------------------
with tab2:

    st.set_page_config(page_title="EpiDemic - COVID-19 Prediction", page_icon="🦠", layout="wide")
    st.title("🦠 COVID-19 Outbreak Prediction")
    st.image("images/coronavirus.jpg", width=800)
    st.write("Predict the likelihood of a COVID-19 outbreak with an interactive and fun form! 😷")
    
    # ---------------------- USER INPUTS ----------------------
    st.header("👤 Fill Your Information")
    
    age = st.slider("👶 Age", 0, 120, 30)
    gender = st.radio("Gender", ["👨 Male", "👩 Female"])
    symptoms = st.multiselect(
        "Select your symptoms 🤒",
        ['Fever 🌡️', 'Cough 🤧', 'Fatigue 😴', 'Loss of Taste/Smell 👅', 'Shortness of Breath 😤']
    )
    travel_history = st.radio("Recent Travel History?", ["🛫 Yes", "🏠 No"])
    contact_with_infected = st.radio("Contact with Infected Person?", ["🤝 Yes", "🙅 No"])
    test_indication = st.selectbox("Reason for Test?", ["Contact with Infected", "Travel History", "Symptoms"])
    
    # ---------------------- ENCODE INPUTS ----------------------
    age_60_and_above = 1 if age >= 60 else 0
    gender_val = 1 if gender.startswith("👨") else 0
    test_map = {"Contact with Infected":0, "Travel History":1, "Symptoms":2}
    test_indication_val = test_map[test_indication]
    
    input_data_covid = pd.DataFrame({
        'cough':[1 if 'Cough 🤧' in symptoms else 0],
        'fever':[1 if 'Fever 🌡️' in symptoms else 0],
        'sore_throat':[0],  # default for demo
        'shortness_of_breath':[1 if 'Shortness of Breath 😤' in symptoms else 0],
        'head_ache':[0],
        'age_60_and_above':[age_60_and_above],
        'gender':[gender_val],
        'test_indication':[test_indication_val]
    })
    
    # ---------------------- LOAD MODELS ----------------------
    algonames_covid = ['Logistic Regression', 'Random Forest', 'XGBoost']
    modelnames_covid = ['covid_logistic.pkl', 'covid_random.pkl', 'covid_xgb.pkl']
    predictions_covid = []
    
    def predict_covid(data):
        predictions_covid.clear()
        for modelname in modelnames_covid:
            with open(modelname,'rb') as f:
                model = pickle.load(f)
            pred = model.predict(data)
            predictions_covid.append(pred)
        return predictions_covid
    
    
    # ---------------------- PREDICTION BUTTON ----------------------
    if st.button("Predict COVID-19 Risk"):
        st.write("🔍 Analyzing your inputs...")
        my_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.01)
            my_bar.progress(percent + 1)
            
        results_covid = predict_covid(input_data_covid)
        
        st.subheader("Prediction Results:")
        st.markdown('-----------------')
        
        # ✅ Save predictions to session_state correctly
        if 'covid_predictions' not in st.session_state:
            st.session_state.covid_predictions = pd.DataFrame(columns=['Model','Prediction'])
        
        for i in range(len(results_covid)):
            model_name = algonames_covid[i]
            prediction_label = "High Risk" if results_covid[i][0]==0 else "Low Risk"
            
            st.subheader(f"Using {model_name}:")
            if results_covid[i][0] == 0:
                st.write("🔥 High Risk")
                st.image("images/covid-positive.jpg", width=250)
            else:
                st.write("🛡️ Low Risk")
                st.image("images/covid-negative.jpg", width=250)
            st.markdown('-----------------')
            
            # Append to session_state
            st.session_state.covid_predictions = pd.concat([
                st.session_state.covid_predictions,
                pd.DataFrame({'Model':[model_name], 'Prediction':[prediction_label]})
            ], ignore_index=True)

#---------------------- H1N1 FLU VACCINE PREDICTION TAB ----------------------

with tab3:    
        
        # ---------------------- APP TITLE ----------------------
        st.set_page_config(page_title="H1N1 Flu Vaccine Prediction", page_icon="💉", layout="wide")
        st.title("💉 H1N1 Flu Vaccine Prediction")
        st.image("images/h1n1.jpg", width=800)
        st.write("This interactive  predicts whether a person is likely to get the H1N1 flu vaccine based on various health and behavioral factors. 🦠")
        
        # ---------------------- USER INPUTS ----------------------
        st.header("👤 Fill Your Information")
        
        # Example H1N1 inputs (you can expand to all features used in your model)
        h1n1_concern = st.slider("H1N1 Concern (0-4) 🤔", 0, 4, 2)
        h1n1_knowledge = st.slider("H1N1 Knowledge (0-3) 📚", 0, 3, 1)
        behavioral_antiviral_meds = st.radio("Used Antiviral Medications? 💊", ["Yes", "No"])
        behavioral_avoidance = st.radio("Avoid Large Gatherings? 👥", ["Yes", "No"])
        behavioral_face_mask = st.radio("Use Face Mask? 😷", ["Yes", "No"])
        behavioral_wash_hands = st.radio("Wash Hands Frequently? 🧼", ["Yes", "No"])
        behavioral_large_gatherings = st.radio("Attend Large Gatherings? 🏟️", ["Yes", "No"])
        behavioral_outside_home = st.radio("Spend Time Outside Home? 🏠", ["Yes", "No"])
        behavioral_touch_face = st.radio("Touch Face Frequently? 🤲", ["Yes", "No"])
        doctor_recc_h1n1 = st.radio("Doctor Recommended H1N1 Vaccine? 👨‍⚕️", ["Yes", "No"])
        doctor_recc_seasonal = st.radio("Doctor Recommended Seasonal Vaccine? 👩‍⚕️", ["Yes", "No"])
        chronic_med_condition = st.radio("Have Chronic Medical Condition? 🏥", ["Yes", "No"])
        child_under_6_months = st.radio("Child Under 6 Months in Household? 👶", ["Yes", "No"])
        health_worker = st.radio("Are you a Health Worker? 🏨", ["Yes", "No"])
        health_insurance = st.radio("Have Health Insurance? 🏥", ["Yes", "No"])
        opinion_h1n1_vacc_effective = st.slider("H1N1 Vaccine Effectiveness Opinion (0-4) 💡", 0, 4, 2)
        opinion_h1n1_risk = st.slider("H1N1 Risk Opinion (0-4) ⚠️", 0, 4, 2)
        opinion_h1n1_sick_from_vacc = st.slider("Sick From H1N1 Vaccine Opinion (0-4) 🤢", 0, 4, 2)
        opinion_seas_vacc_effective = st.slider("Seasonal Vaccine Effectiveness Opinion (0-4) 💉", 0, 4, 2)
        opinion_seas_risk = st.slider("Seasonal Risk Opinion (0-4) ⚠️", 0, 4, 2)
        opinion_seas_sick_from_vacc = st.slider("Sick From Seasonal Vaccine Opinion (0-4) 🤮", 0, 4, 2)
        age_group = st.selectbox("Age Group", [1, 2, 3, 4, 5])
        education = st.selectbox("Education Level", [1, 2, 3, 4])
        race = st.selectbox("Race", [1, 2, 3, 4])
        sex = st.radio("Sex", ["Male", "Female"])
        income_poverty = st.selectbox("Income Level", [1, 2, 3, 4])
        marital_status = st.radio("Marital Status", ["Married", "Single"])
        rent_or_own = st.radio("Rent or Own Home?", ["Rent", "Own"])
        employment_status = st.selectbox("Employment Status", [0, 1])
        hhs_geo_region = st.selectbox("HHS Geo Region", list(range(1, 11)))
        census_msa = st.selectbox("Census MSA", [1, 2, 3])
        household_adults = st.number_input("Number of Adults in Household", 0, 10, 2)
        household_children = st.number_input("Number of Children in Household", 0, 10, 1)
        employment_industry = st.selectbox("Employment Industry", [1, 2, 3, 4])
        employment_occupation = st.selectbox("Employment Occupation", [1, 2, 3, 4])
        seasonal_vaccine = st.radio("Took Seasonal Vaccine? 💉", ["Yes", "No"])
        
        # ---------------------- ENCODE INPUTS ----------------------
        yes_no_map = {"Yes":1, "No":0, "Male":1, "Female":0, "Married":1, "Single":0, "Own":1, "Rent":0}
        
        input_data_h1n1 = pd.DataFrame({
            'h1n1_concern':[h1n1_concern],
            'h1n1_knowledge':[h1n1_knowledge],
            'behavioral_antiviral_meds':[yes_no_map[behavioral_antiviral_meds]],
            'behavioral_avoidance':[yes_no_map[behavioral_avoidance]],
            'behavioral_face_mask':[yes_no_map[behavioral_face_mask]],
            'behavioral_wash_hands':[yes_no_map[behavioral_wash_hands]],
            'behavioral_large_gatherings':[yes_no_map[behavioral_large_gatherings]],
            'behavioral_outside_home':[yes_no_map[behavioral_outside_home]],
            'behavioral_touch_face':[yes_no_map[behavioral_touch_face]],
            'doctor_recc_h1n1':[yes_no_map[doctor_recc_h1n1]],
            'doctor_recc_seasonal':[yes_no_map[doctor_recc_seasonal]],
            'chronic_med_condition':[yes_no_map[chronic_med_condition]],
            'child_under_6_months':[yes_no_map[child_under_6_months]],
            'health_worker':[yes_no_map[health_worker]],
            'health_insurance':[yes_no_map[health_insurance]],
            'opinion_h1n1_vacc_effective':[opinion_h1n1_vacc_effective],
            'opinion_h1n1_risk':[opinion_h1n1_risk],
            'opinion_h1n1_sick_from_vacc':[opinion_h1n1_sick_from_vacc],
            'opinion_seas_vacc_effective':[opinion_seas_vacc_effective],
            'opinion_seas_risk':[opinion_seas_risk],
            'opinion_seas_sick_from_vacc':[opinion_seas_sick_from_vacc],
            'age_group':[age_group],
            'education':[education],
            'race':[race],
            'sex':[yes_no_map[sex]],
            'income_poverty':[income_poverty],
            'marital_status':[yes_no_map[marital_status]],
            'rent_or_own':[yes_no_map[rent_or_own]],
            'employment_status':[employment_status],
            'hhs_geo_region':[hhs_geo_region],
            'census_msa':[census_msa],
            'household_adults':[household_adults],
            'household_children':[household_children],
            'employment_industry':[employment_industry],
            'employment_occupation':[employment_occupation],
            'seasonal_vaccine':[yes_no_map[seasonal_vaccine]]
        })
        
        # ---------------------- LOAD MODELS ----------------------
        algonames_h1n1 = ['Logistic Regression', 'Random Forest', 'XGBoost']
        modelnames_h1n1 = ['h1n1_logistic.pkl', 'h1n1_random.pkl', 'h1n1_xgb.pkl']
        predictions_h1n1 = []
        
        def predict_h1n1(data):
            predictions_h1n1.clear()
            for modelname in modelnames_h1n1:
                with open(modelname,'rb') as f:
                    model = pickle.load(f)
                pred = model.predict(data)
                predictions_h1n1.append(pred)
            return predictions_h1n1
        import time

        # ---------------------- H1N1 PREDICTION BUTTON ----------------------
        if st.button("Predict H1N1 Vaccination Likelihood"):
            st.write("🔍 Analyzing your inputs...")
            my_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)
                my_bar.progress(percent + 1)
                
            # Make predictions using all models
            results_h1n1 = predict_h1n1(input_data_h1n1)
            
            st.subheader("Prediction Results:")
            st.markdown('-----------------')
            
            # ✅ Save predictions to session_state correctly
            if 'h1n1_predictions' not in st.session_state:
                st.session_state.h1n1_predictions = pd.DataFrame(columns=['Model','Prediction'])
            
            for i in range(len(results_h1n1)):
                model_name = algonames_h1n1[i]
                # Label based on prediction (adjust if your model outputs 0/1 differently)
                prediction_label = "Likely" if results_h1n1[i][0]==1 else "Unlikely"
                
                st.subheader(f"Using {model_name}:")
                if results_h1n1[i][0] == 1:
                    st.write("💉 Likely")
                    st.image("images/Screenshot 2025-11-09 021651.png", width=250)
                else:
                    st.write("❌ Unlikely")
                    st.image("images/Screenshot 2025-11-09 022124.png", width=250)
                st.markdown('-----------------')
                
                # Append to session_state
                st.session_state.h1n1_predictions = pd.concat([
                    st.session_state.h1n1_predictions,
                    pd.DataFrame({'Model':[model_name], 'Prediction':[prediction_label]})
                ], ignore_index=True)


with tab4:
        
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        
        st.set_page_config(page_title="EpiDemic Dashboard", page_icon="📊", layout="wide")
        st.title("📊 EpiDemic Prediction Dashboard")
        
        
        # ---------------------- DATA STORAGE ----------------------
        if 'covid_predictions' not in st.session_state:
            st.session_state.covid_predictions = pd.DataFrame(columns=['Model','Prediction'])
        
        if 'h1n1_predictions' not in st.session_state:
            st.session_state.h1n1_predictions = pd.DataFrame(columns=['Model','Prediction'])
        
        # ---------------------- COVID DASHBOARD ---------------------
        st.subheader("🦠 COVID-19 Predictions")
        # Paste the COVID grouped bar chart code here
        if 'covid_predictions' in st.session_state and not st.session_state.covid_predictions.empty:
            covid_data = st.session_state.covid_predictions.copy()
            covid_counts = covid_data.groupby(['Model','Prediction']).size().reset_index(name='Count')
            fig = px.bar(
                covid_counts,
                x='Model',
                y='Count',
                color='Prediction',
                barmode='group',
                color_discrete_map={'High Risk':'red', 'Low Risk':'green'},
                title="COVID-19 Predictions by Model"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No COVID predictions yet. Make predictions in the COVID tab.")
    
        st.markdown("---")
    
        st.subheader("💉 H1N1 Predictions")

        # The H1N1 grouped bar chart code here

        if 'h1n1_predictions' in st.session_state and not st.session_state.h1n1_predictions.empty:
            h1n1_data = st.session_state.h1n1_predictions.copy()
            h1n1_counts = h1n1_data.groupby(['Model','Prediction']).size().reset_index(name='Count')
            fig2 = px.bar(
                h1n1_counts,
                x='Model',
                y='Count',
                color='Prediction',
                barmode='group',
                color_discrete_map={'Likely':'green','Unlikely':'red'},
                title="H1N1 Predictions by Model"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No H1N1 predictions yet. Make predictions in the H1N1 tab.")
    