import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from streamlit_option_menu import option_menu
import time

# --- إعدادات الصفحة وتنسيق الواجهة ---
st.set_page_config(page_title="نظام الكاشير الذكي", layout="wide", page_icon="🛒")

# --- تنسيق CSS مخصص لجعل الموقع جذاب ويدعم العربية ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    * {
        font-family: 'Cairo', sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    /* تنسيق الكروت */
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        text-align: center;
        border: 1px solid #4e4e4e;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: scale(1.05);
        border-color: #ff4b4b;
    }
    
    /* تنسيق العناوين */
    h1, h2, h3 {
        color: #ffffff;
        text-align: right;
    }
    
    /* جعل النصوص من اليمين لليسار */
    .element-container, .stMarkdown, .stTextInput, .stNumberInput {
        direction: rtl;
        text-align: right;
    }
    
    /* تحسين شكل الأزرار */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
    }
    
    /* تأثيرات حركية */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .animate-text {
        animation: fadeIn 1.5s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# --- الاتصال بجوجل شيت ---
@st.cache_resource
def connect_to_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Shop_System") # تأكد من أن اسم الملف في جوجل درايف مطابق لهذا الاسم
        return sheet
    except Exception as e:
        st.error(f"⚠️ خطأ في الاتصال بقاعدة البيانات: {e}")
        return None

# دالة لجلب البيانات (مع التخزين المؤقت لتسريع الأداء)
def get_data(sheet_object, worksheet_name):
    worksheet = sheet_object.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

sheet = connect_to_gsheet()

# --- القائمة الجانبية ---
with st.sidebar:
    selected = option_menu(
        menu_title="القائمة الرئيسية",
        options=["شاشة البيع", "إضافة منتج", "المخزون"],
        icons=["cart-check", "plus-circle", "database"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#262730"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "18px", "text-align": "right", "margin":"0px", "--hover-color": "#333"},
            "nav-link-selected": {"background-color": "#ff4b4b"},
        }
    )
    st.markdown("---")
    st.caption("🚀 نظام متطور v1.0")

# ==========================
# 1. شاشة البيع (POS)
# ==========================
if selected == "شاشة البيع":
    st.markdown("<h1 class='animate-text'>🛒 نقطة البيع الذكية</h1>", unsafe_allow_html=True)
    
    if sheet:
        # تحميل المخزون
        inventory_worksheet = sheet.worksheet("Inventory")
        inventory_data = inventory_worksheet.get_all_records()
        df_inv = pd.DataFrame(inventory_data)
        
        # التأكد من أن الأعمدة رقمية
        if not df_inv.empty:
            df_inv['Barcode'] = df_inv['Barcode'].astype(str)
            df_inv['Price'] = pd.to_numeric(df_inv['Price'], errors='coerce')
            df_inv['Quantity'] = pd.to_numeric(df_inv['Quantity'], errors='coerce')

        # تقسيم الشاشة
        col_scan, col_details = st.columns([1, 2])
        
        product_found = None
        
        with col_scan:
            st.info("💡 قم بمسح الباركود أو كتابته واضغط Enter")
            barcode_input = st.text_input("باركود المنتج", key="barcode_scanner", placeholder="Scan here...", help="ضع المؤشر هنا واستخدم قارئ الباركود")

        # منطق البحث عن المنتج
        if barcode_input and not df_inv.empty:
            product_found = df_inv[df_inv['Barcode'] == barcode_input]
            
            if not product_found.empty:
                product_data = product_found.iloc[0]
                
                with col_details:
                    # عرض تفاصيل المنتج بشكل جذاب
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="color: #ff4b4b; margin:0;">{product_data['Name']}</h2>
                        <h4 style="color: #ccc;">النوع: {product_data['Type']}</h4>
                        <hr>
                        <h1 style="color: #4CAF50;">{product_data['Price']:,.2f} EGP</h1>
                        <p>الكمية المتاحة بالمخزن: <b>{product_data['Quantity']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("---")
                
                # منطقة إتمام عملية البيع
                c1, c2, c3 = st.columns(3)
                with c1:
                    qty_to_buy = st.number_input("الكمية المطلوبة", min_value=1, max_value=int(product_data['Quantity']), value=1, step=1)
                
                total_price = qty_to_buy * product_data['Price']
                
                with c2:
                    st.metric(label="الإجمالي المطلوب دفعه", value=f"{total_price:,.2f} EGP")
                
                with c3:
                    st.write("##") # مسافة
                    confirm_btn = st.button("✅ إتمام البيع", type="primary")
                
                if confirm_btn:
                    if qty_to_buy <= product_data['Quantity']:
                        with st.spinner("جاري تسجيل العملية..."):
                            # 1. تحديث الكمية في المخزون
                            cell = inventory_worksheet.find(barcode_input)
                            current_qty = int(product_data['Quantity'])
                            new_qty = current_qty - qty_to_buy
                            inventory_worksheet.update_cell(cell.row, 5, new_qty) # العمود 5 هو الكمية
                            
                            # 2. تسجيل البيع في شيت المبيعات
                            sales_worksheet = sheet.worksheet("Sales")
                            sales_worksheet.append_row([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                product_data['Name'],
                                qty_to_buy,
                                total_price,
                                "تم الدفع"
                            ])
                            
                            st.balloons()
                            st.toast(f"تم بيع {qty_to_buy} من {product_data['Name']} بنجاح!", icon="🎉")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("الكمية المطلوبة غير متاحة في المخزون!")
            else:
                st.warning("❌ المنتج غير موجود! تأكد من الباركود أو قم بتسجيله أولاً.")
        elif barcode_input:
             st.warning("المخزون فارغ أو حدث خطأ.")

# ==========================
# 2. إضافة منتج جديد
# ==========================
elif selected == "إضافة منتج":
    st.markdown("<h1 class='animate-text'>📦 تسجيل منتج جديد</h1>", unsafe_allow_html=True)
    
    with st.form("add_product_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            new_barcode = st.text_input("الباركود (Barcode)")
            new_name = st.text_input("اسم المنتج")
            new_type = st.selectbox("النوع / التصنيف", ["عام", "إلكترونيات", "ملابس", "أغذية", "أخرى"])
        
        with c2:
            new_price = st.number_input("سعر البيع", min_value=0.0, step=0.5)
            new_qty = st.number_input("الكمية الأولية", min_value=1, step=1)
            # يمكن إضافة سعر التكلفة مستقبلاً
        
        submitted = st.form_submit_button("💾 حفظ البيانات في المخزون")
        
        if submitted:
            if new_barcode and new_name:
                if sheet:
                    inventory_worksheet = sheet.worksheet("Inventory")
                    # التحقق من عدم التكرار
                    try:
                        existing = inventory_worksheet.find(new_barcode)
                        st.error("⚠️ هذا الباركود مسجل مسبقاً!")
                    except gspread.exceptions.CellNotFound:
                        inventory_worksheet.append_row([new_barcode, new_name, new_type, new_price, new_qty])
                        st.success(f"تم إضافة {new_name} للمخزون بنجاح!")
                        st.balloons()
            else:
                st.warning("يرجى ملء الباركود واسم المنتج على الأقل.")

# ==========================
# 3. عرض المخزون (للمتابعة)
# ==========================
elif selected == "المخزون":
    st.markdown("<h1 class='animate-text'>📊 حالة المخزون الحالية</h1>", unsafe_allow_html=True)
    
    if sheet:
        df = get_data(sheet, "Inventory")
        if not df.empty:
            # تلوين الجدول
            st.dataframe(df.style.highlight_max(axis=0, color='darkgreen'), use_container_width=True)
            
            st.markdown("### إحصائيات سريعة")
            k1, k2, k3 = st.columns(3)
            k1.metric("عدد الأصناف", len(df))
            
            # تحويل الأعمدة لأرقام للعمليات الحسابية
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)

            k2.metric("إجمالي القطع", int(df['Quantity'].sum()))
            total_value = (df['Quantity'] * df['Price']).sum()
            k3.metric("القيمة التقديرية للمخزون", f"{total_value:,.2f} EGP")
        else:
            st.info("المخزون فارغ حالياً.")
