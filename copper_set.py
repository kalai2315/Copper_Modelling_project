import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu




def predict_status(quantity_tons, customer, country, item_type,
                   application, thickness, width, product_ref, selling_price,
                   item_day, item_month, item_year, delivery_year,
                   delivery_month, delivery_day):
    # Load the classification model
    with open("F:/project/Copper_set/Classification_model.pkl", "rb") as f1:
        class_model = pickle.load(f1)

    # Define the new sample as a 2D numpy array
    new_sample = np.array([[quantity_tons, customer, country, item_type,
                             application, thickness, width, product_ref, selling_price,
                             item_day, item_month, item_year, delivery_year,
                             delivery_month, delivery_day]])

    # Predict the status using the loaded model
    new_predict = class_model.predict(new_sample)

    if new_predict == 1:
        return "Won"
    else:
        return "Lose"

def predict_selling_price(quantity_tons, customer, country, status, item_type,
                          application, thickness, width, product_ref,
                          item_day, item_month, item_year, delivery_year,
                          delivery_month, delivery_day):
    # Load the regression model
    with open(r'F:\project\Copper_set\regression_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Define the user data as a 2D numpy array
    user_data = np.array([[quantity_tons, customer, country, status, item_type,
                          application, thickness, width, product_ref,
                          item_day, item_month, item_year, delivery_year,
                          delivery_month, delivery_day]])

    # Predict the selling price using the loaded model
    y_pred = model.predict(user_data)

    # Convert the predicted log-transformed selling price back to the original scale
    selling_price = np.exp(y_pred[0])

    # Return the predicted selling price
    return selling_price

# Streamlit page custom design
st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["PREDICT STATUS", "PREDICT SELLING PRICE"])

if tab1:
    st.header("PREDICT STATUS (Won / Lose)")
    st.write(" ")

    col1, col2 = st.columns(2)

    with col1:
        quantity_tons = st.number_input(label="**Enter the Value for QUANTITY_TONS**/ Min:0.0, Max:11.15", format="%0.15f", key="quantity_tons_tab1")
        customer = st.number_input(label="**Enter the Value for CUSTOMER**/ Min:12458.0, Max:2147483647.0", format="%0.15f", key="customer_tab1")
        country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0", key="country_tab1")
        item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0", key="item_type_tab1")
        application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:99", key="application_tab1")
        thickness = st.number_input(label="**Enter the Value for THICKNESS **/ Min:0.1655, Max: 7.8244", format="%0.15f", key="thickness_tab1")
        width = st.number_input(label="**Enter the Value for WIDTH**/ Min:0.69, Max:8", key="width_tab1")
        product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579", key="product_ref_tab1")

    with col2:
        selling_price = st.number_input(label="**Enter the Value for SELLING PRICE (Log Value)**/ Min:0.095, Max:11.30", format="%0.15f", key="selling_price_tab1")
        item_day = st.selectbox("**Select the Day for ITEM DATE**", tuple(range(1, 32)), key="item_day_tab1")
        item_month = st.selectbox("**Select the Month for ITEM DATE**", tuple(range(1, 13)), key="item_month_tab1")
        item_year = st.selectbox("**Select the Year for ITEM DATE**", ("2020", "2021"), key="item_year_tab1")
        delivery_day = st.selectbox("**Select the Day for DELIVERY DATE**", tuple(range(1, 32)), key="delivery_day_tab1")
        delivery_month = st.selectbox("**Select the Month for DELIVERY DATE**", tuple(range(1, 13)), key="delivery_month_tab1")
        delivery_year = st.selectbox("**Select the Year for DELIVERY DATE**", ("2020", "2021", "2022"), key="delivery_year_tab1")

    button = st.button(":yellow[***PREDICT THE STATUS***]", use_container_width=True)

    if button:
        status = predict_status(quantity_tons, customer, country, item_type,
                             application, thickness, width, product_ref, selling_price,
                             item_day, item_month, item_year, delivery_year,
                             delivery_month, delivery_day)

        if status == "Won":
            st.write("## :green[**The Status is WON**]")
        else:
            st.write("## :red[**The Status is LOSE**]")

if tab2:
    st.header("Predict Selling Price")
    st.write(" ")

    col1, col2 = st.columns(2)

    with col1:
        quantity_tons = st.number_input(label="**Enter the Value for QUANTITY_TONS**/ Min:0.0, Max:11.15", format="%0.15f")
        customer = st.number_input(label="**Enter the Value for CUSTOMER**/ Min:12458.0, Max:2147483647.0", format="%0.15f")
        country = st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        item_type = st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application = st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:99")
        thickness = st.number_input(label="**Enter the Value for THICKNESS **/ Min:0.1655, Max: 7.8244", format="%0.15f")
        width = st.number_input(label="**Enter the Value for WIDTH**/ Min:0.69, Max:8")
        product_ref = st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")

    with col2:
        status= st.number_input(label="**Enter the Value for STATUS**/ Min:0.0, Max:1.0")
        item_day = st.selectbox("**Select the Day for ITEM DATE**", tuple(range(1, 32)))
        item_month = st.selectbox("**Select the Month for ITEM DATE**", tuple(range(1, 13)))
        item_year = st.selectbox("**Select the Year for ITEM DATE**", ("2020", "2021"))
        delivery_day = st.selectbox("**Select the Day for DELIVERY DATE**", tuple(range(1, 32)))
        delivery_month = st.selectbox("**Select the Month for DELIVERY DATE**", tuple(range(1, 13)))
        delivery_year = st.selectbox("**Select the Year for DELIVERY DATE**", ("2020", "2021", "2022"))

    button = st.button(":yellow[***PREDICT THE SELLING PRICE***]", use_container_width=True)

    if button:
        selling_price = predict_selling_price(quantity_tons, customer, country, status, item_type,
                                               application, thickness, width, product_ref,
                                               item_day, item_month, item_year, delivery_year,
                                               delivery_month, delivery_day)
        st.write("## :green[**The Selling Price is :**]",selling_price)