import streamlit as st
import pandas as pd
import os
import requests
import csv
import json
import shutil
import re
from datetime import date
from urllib.parse import urlparse
from openai import OpenAI
from PIL import Image, ImageOps
from io import BytesIO
import zipfile
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed



def get_output_folder(base_name):
    today_date = date.today().isoformat()
    version = 1

    while True:
        folder_name = f"{base_name}_{today_date}_{version}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
            return folder_name
        version += 1
        
        
# Initialisation
today = date.today().isoformat()
output_base_dir = "output_folder"
output_dir = get_output_folder(output_base_dir)
attribute_base_dir = os.path.join(output_dir, "attributes")
os.makedirs(attribute_base_dir, exist_ok=True)

load_dotenv()


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

images_zip_path = os.path.join(output_dir, "all_images.zip")
images_csv_path = os.path.join(output_dir, "structured_images.csv")
image_data = []

languages = {
    2: "en",
    3: "fr",
    4: "es",
    5: "pt",
    6: "it",
    7: "de",
    8: "nl"
}

fieldnames_price = ["sku", "website_id", "price", "special_price"]
fieldnames_status = ["sku", "store_id", "status"]
fieldnames_missing_content = ["sku", "onedirect_warranty_time", "store_id"]
fieldnames_processed = ["sku", "product name", "Brand", "PanNumber", "Store", "Price", "special_price", "attribut_set"]
fieldnames_openai = ["sku", "store_id","name", "onedirect_baseline", "description", "short_description", "visibility"]
fieldnames_consolidated = ["sku", "ean", "PanNumber", "Brand", "attribute_set_code"] + [f"img.{i}" for i in range(1, 31)]

# Fieldnames for attributes
fieldnames_headsets = ["GTIN","PanNumber","Brand","sku","store_id","od_ces_cable_lenght","od_connectivity","od_headset_type","od_ces_microphone_boom","od_ces_mono_duo","od_ces_mute","od_ces_noise_cancelling","od_ces_range","od_ces_recharge_time","od_ces_standby_time_talk_time","od_ces_wearing_style","od_ces_weight"]
fieldnames_walkie = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_tws_frequency", "od_ip_rating", "od_tws_channels_sub_channels_c", "od_tws_battery", "od_tws_range", "od_battery_type", "od_tws_vibrate_function", "od_size", "od_pack"]
fieldnames_video = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_video_type", "od_vce_connects_to_use_with", "od_operating_system", "od_vce_camera_resolution", "od_size"]
fieldnames_screens = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_use_type", "od_screen_size", "od_operational_hours", "od_brightness", "od_other_features"]
fieldnames_mobile = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_mobile_network", "od_tme_dual_sim", "od_tme_gps", "od_tme_operating_system", "od_tme_nfc", "od_tme_dust_shock_resistant", "od_militar_standard", "od_ip_rating", "od_tme_camera", "od_tme_rear_camera", "od_tme_front_camera", "od_tme_bluetooth", "od_tme_standby_talk_time", "od_tme_weight", "od_tme_das", "od_tme_gsm_network"]
fieldnames_corded = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_tfs_number_of_lines", "od_sip_accounts", "od_colour", "od_tfs_function_keys", "od_tfs_phonebook_features", "od_tfs_expansion_module", "od_tfs_speakerphone", "od_tfs_headset_port", "od_tfs_power_source", "od_display", "od_tfs_call_indicator_led", "od_tfs_connectivity", "od_tfs_wall_mounting"]
fieldnames_camera = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_wam_resolution", "od_fps", "od_vce_camera_field_of_view"]
fieldnames_audio = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_phone_system", "od_other_features", "od_size"]
fieldnames_cordless = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_tsf_range", "od_display", "od_display", "od_tsf_caller_id", "od_tsf_call_recording", "od_tsf_contacts", "od_tsf_vip_groups", "od_tsf_bluetooth", "od_tsf_headset_port", "od_tsf_sd_card_slot", "od_tsf_sms", "od_tsf_vibrate", "od_ip_rating", "od_tsf_remote_base", "od_tsf_additional_handsets_sup"]
fieldnames_tablet = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_mobile_network", "od_tme_dual_sim", "od_tme_gps", "od_tme_operating_system", "od_tme_nfc", "od_tme_dust_shock_resistant", "od_militar_standard", "od_ip_rating", "od_tme_camera", "od_tme_rear_camera", "od_tme_front_camera", "od_tme_bluetooth", "od_tme_screen_size", "od_tme_standby_talk_time", "od_tme_weight"]
fieldnames_content_sharing = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_presentation_type", "od_screen_sharing", "od_sharing_options", "od_sharing_works_with", "od_sharing_connections", "od_resolution", "od_sharing_touch_screen", "od_sharing_buttons", "od_sharing_features", "od_connectivity", "od_size"]
fieldnames_accessory = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_accessory_type"]
fieldnames_network = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_network_product_type", "od_sch_managed", "od_sch_rackable", "od_sch_power_over_ethernet_poe", "od_sch_poe_ports", "od_sch_poe", "od_sch_lan_capacity", "od_sch_ethernet_ports", "od_sch_gigabit", "od_sch_gigabit_ports", "od_sch_sfp_ports", "od_sch_ipv6", "od_sch_qos"]
fieldnames_standard = ["GTIN", "PanNumber", "Brand", "sku", "store_id", "od_line_type", "od_sip_accounts", "od_tsf_call_recording", "od_tsf_additional_handsets_sup", "od_tsf_contacts", "od_tfs_main_filters", "od_size"]


writers = {}
unique_skus = set()
attribute_writers = {}
consolidated_file_path = os.path.join(output_dir, f"{today}_consolidated_product_list.csv")


def fetch_product_data(row):
    """Fetch product data from Icecat API."""
    # Récupération de la clé depuis Streamlit secrets
    icecat_key = st.secrets["ICECAT_API_TOKEN"]
    headers = {
        "api-token": icecat_key
    }
    # Construction de l'URL de l'API Icecat
    url = (
        "https://live.icecat.biz/api"
        f"?UserName=Patricel"
        f"&lang={languages[row['Store']]}"
        f"&Brand={row['Brand']}"
        f"&ProductCode={row['PanNumber']}"
    )

    # Requête à l'API
    response = requests.get(url, headers=headers)

    # Vérification du statut de la requête
    if response.status_code == 200:
        return response.json()
    else:
        # Log ou gestion d’erreur 
        st.warning(f"Erreur HTTP {response.status_code} lors de la récupération des données.")
        return None


def sanitize_broken_json_string(content):
    """Nettoie un JSON sous forme de string contenant des erreurs classiques dans les valeurs."""
        
    content = content.strip().replace("```json", "").replace("```", "")
    content = content.replace("“", "''").replace("”", "''")  # guillemets typographiques

    # Match tous les champs de type "clé": "valeur" (même sur plusieurs lignes), avec une tolérance au contenu foireux
    def fix_quoted_values(match):
        key = match.group(1)
        raw_val = match.group(2)
        # échappe les guillemets internes mal échappés
        fixed_val = re.sub(r'(?<!\\)"', r"''", raw_val)
        fixed_val = raw_val.replace('\\"', "''")
        fixed_val = re.sub(r'(?<!\\)"', "''", fixed_val)
        return f'"{key}": "{fixed_val}"'

    # Expression régulière pour capturer chaque paire clé:valeur textuelle
    pattern = r'"([^"]+)":\s*"(.*?)"(?=\s*[,}])'  # match non-greedy sur la valeur
    fixed = re.sub(pattern, fix_quoted_values, content, flags=re.DOTALL)

    return fixed

def extract_images(data):
    """Extract image URLs from API data."""
    images = []
    if 'Image' in data and 'HighPic' in data['Image']:
        images.append(data['Image']['HighPic'])
    
    if 'Gallery' in data:
        for gallery_item in data['Gallery']:
            if 'Pic' in gallery_item:
                images.append(gallery_item['Pic'])
    
    return images


def generate_openai_content(api_data, row, url):
    languages_for_prompt = { 
        2: "en - English",
        3: "fr - Français",
        4: "es - Español",
        5: "pt - Português",
        6: "it - Italiano",
        7: "de - Deutsch",
        8: "nl - Nederlands"
    }

    language_label = languages_for_prompt.get(row["Store"], "en - English")
    """Generate content using OpenAI API."""
    ai_prompt = """
You MUST respond in {3}, meaning the content should be written entirely in that language.
You are an experienced SEO copywriter working for the {3} website, specialized in tech and telecom products. Your goal is to deliver a clear, persuasive product description using simple HTML tags only: <h2>, <p>, <strong>, <em>, <ul>, <li>, <table>, <tr>, <td>.

You MUST write in {3}.
Follow the latest European spelling and punctuation conventions for that language.
Use short sentences, active voice, and aim for a high Flesch Reading Ease score. Never mention any brand other than the product’s own or the name of the shop.

The product information is available in the following JSON: {1}
The product name is: {2}

Please generate the following, as a valid JSON object with properly escaped quotes (\\") and Do not include raw line breaks in JSON strings. Keep all HTML content on a single line.:

1. name — a short version of the product name (max 60 characters)

2. baseline — a catchy sentence 

3. features — a <ul> with exactly 5 <li> items.
   Each <li> starts with a <strong>concise benefit</strong> followed by a short explanation.
   This will be used as the short description.

4. description — a full HTML product description structured into flowing <p>, <ul>, <table> blocks. Include <strong> for keypoint of the desciption (but **without any section titles like <h3>**) Include:
   - A short title <H3> including the name of the product
   - A product overview paragraph. (Several sentences introducing the products main value inside <p> tags.)
   - A longer and more detailed version of the 5 benefits from the "features" section, expanded as natural text.(detail each point of the features)
   - Real-world usage examples and advantages. (A few <p> paragraphs describing real-world usage and advantages.)
   - A <table> with up to 8 relevant technical details (2 columns: label and value).
   - A persuasive final call-to-action.

5. weight — extracted from the product data, if available

6. width — extracted if available

7. height — extracted if available

8. depth — extracted if available

If the model fails to understand the request or data is missing, return an empty JSON.
"""

    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY )
    if not openai_key:
        st.error("Clé API OpenAI non configurée.")
        return None

    client = OpenAI(api_key=openai_key)

    try:
        infos = {
            "GeneralInfo": api_data["data"]["GeneralInfo"],
            "FeaturesGroups": api_data["data"]["FeaturesGroups"]
        }
        title = api_data["data"]["GeneralInfo"]["Title"]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": ai_prompt.format(url, infos, title, language_label),
                    "max_tokens": 6000
                }
            ]
        )

        # Debug: Log the raw response
        # st.write("Raw OpenAI Response:", response)

        # Attempt to parse the response
        if response.choices and response.choices[0].message.content:
            raw_content = response.choices[0].message.content
            cleaned_content = sanitize_broken_json_string(raw_content)

            try:
                return json.loads(cleaned_content)

            except json.JSONDecodeError as json_err:
                st.error(f"❌ Erreur de décode JSON : {json_err}")

                st.text("🔍 Contenu brut retourné par OpenAI :")
                st.code(raw_content, language="json")

                st.text("🔧 Contenu nettoyé avant parsing :")
                st.code(cleaned_content, language="json")

                # ➕ Affiche dans la console terminal (stdout)
                print(f"\n\n[ERREUR JSON - SKU: {row['sku']}]")
                print("=== RAW RESPONSE ===")
                print(raw_content)
                print("=== CLEANED ===")
                print(cleaned_content)

                # ➕ Sauvegarde dans un fichier texte pour analyse ultérieure
                error_log_path = os.path.join(output_dir, "openai_error_log.txt")
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n[SKU: {row['sku']}]\n")
                    f.write("=== RAW RESPONSE ===\n")
                    f.write(raw_content)
                    f.write("\n=== CLEANED ===\n")
                    f.write(cleaned_content)
                    f.write("\n")
                return None

    except Exception as e:
        st.error(f"Erreur avec OpenAI : {str(e)}")
        return None


def validate_image_columns(df):
    required_columns = [f"img.{i}" for i in range(1, 31)]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Les colonnes suivantes sont manquantes dans le fichier : {', '.join(missing_columns)}")
        return False
    return True

def get_attribute_writer(attribute_dir, attribute_file_path, fieldnames):
    """Create a writer for an attribute CSV if not already created."""
    if attribute_file_path not in attribute_writers:
        file = open(attribute_file_path, "w", newline='', encoding='utf-8')
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        attribute_writers[attribute_file_path] = {"writer": writer, "file": file}
    return attribute_writers[attribute_file_path]["writer"]


def process_attributes(row, gtin, country):
    """Process and write attribute data for a specific row."""
    attribut_set = row.get("attribut_set")
    if not attribut_set:
        return

    attribute_dirs = {
        "Headsets": ("Headsets", fieldnames_headsets),
        "Walkie Talkies": ("Walkie Talkies", fieldnames_walkie),
        "Conference - Video": ("Conference Video", fieldnames_video),
        "Conference - Audio": ("Conference Audio", fieldnames_audio),
        "Screens": ("Screens", fieldnames_screens),
        "Phone - Mobile": ("Mobile", fieldnames_mobile),
        "Phone - corded": ("Corded", fieldnames_corded),
        "Cameras - Webcam": ("Camera", fieldnames_camera),
        "Phone - Cordless": ("Cordless", fieldnames_cordless),
        "Tablets": ("Tablet", fieldnames_tablet),
        "Content Sharing": ("Content Sharing", fieldnames_content_sharing),
        "Accessory": ("Accessory", fieldnames_accessory),
        "Network": ("Network", fieldnames_network),
        "Standard": ("Standard", fieldnames_standard),
    }

    if attribut_set in attribute_dirs:
        sub_dir, f_names = attribute_dirs[attribut_set]
        dir_path = os.path.join(attribute_base_dir, sub_dir)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"{today}_{sub_dir.replace(' ', '_').lower()}_{country}.csv")
        writer = get_attribute_writer(dir_path, file_path, f_names)

        attribute_row = {
            "GTIN": gtin,
            "PanNumber": row["PanNumber"],
            "Brand": row["Brand"],
            "sku": row["sku"],
            "store_id": row["Store"]
        }
        # If needed, fill in other attribute fields here before writing
        writer.writerow(attribute_row)


def close_attribute_writers():
    """Close all attribute writer files."""
    for file_info in attribute_writers.values():
        file_info["file"].close()



def download_and_process_image(sku, img_url, index):
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image_square = ImageOps.pad(image, (800, 800), color="white", centering=(0.5, 0.5))
            img_filename = f"{sku}_{index}.jpg"
            img_buffer = BytesIO()
            image_square.save(img_buffer, format="JPEG")
            return {
                "success": True,
                "filename": img_filename,
                "buffer": img_buffer.getvalue(),
                "index": index
            }
        else:
            return {"success": False, "error": f"Échec téléchargement (status {response.status_code})", "index": index}
    except Exception as e:
        return {"success": False, "error": str(e), "index": index}
    

def process_images(df):
    st.info("📸 Début du traitement des images...")
    total_rows = len(df)
    progress_bar = st.progress(0)
    current = 0
    logs = []

    with zipfile.ZipFile(images_zip_path, 'w') as zipf:
        for _, row in df.iterrows():
            sku = row.get("sku")
            additional_images = []
            base_image = small_image = thumbnail_image = ""
            current += 1

            futures = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                for i in range(1, 6):
                    img_url = row.get(f"img.{i}")
                    if pd.notna(img_url) and img_url:
                        futures.append(executor.submit(download_and_process_image, sku, img_url, i))

                results = [f.result() for f in as_completed(futures)]

            # Sort results by image index to ensure correct order
            results.sort(key=lambda r: r["index"])

            for res in results:
                if res["success"]:
                    zipf.writestr(res["filename"], res["buffer"])
                    if res["index"] == 1:
                        base_image = small_image = thumbnail_image = res["filename"]
                    else:
                        additional_images.append(res["filename"])
                else:
                    logs.append(f"❌ {sku} - img.{res['index']} : {res['error']}")

            image_data.append({
                "sku": sku,
                "base_image": base_image,
                "small_image": small_image,
                "thumbnail_image": thumbnail_image,
                "additional_images": ','.join(additional_images)
            })

            progress_bar.progress(current / total_rows)

    st.success("✅ Toutes les images ont été traitées et compressées.")
    pd.DataFrame(image_data).to_csv(images_csv_path, index=False)

    if logs:
        st.warning("⚠️ Quelques erreurs sont survenues :")
        st.code("\n".join(logs), language="text")


    
    
def process_file(file, selected_outputs):
    """Process the uploaded CSV file and generate outputs."""
    df = pd.read_csv(file)
    total = len(df)
    progress_bar = st.progress(0)
    st.info(f"🔢 Nombre de produits à traiter : {total}")
    st.write("Fichier chargé :")
    st.write(df.head())

    # Open consolidated file
    with open(consolidated_file_path, "w", newline='', encoding='utf-8') as consolidated_file:
        consolidated_writer = csv.DictWriter(consolidated_file, fieldnames=fieldnames_consolidated)
        consolidated_writer.writeheader()

        for index, row in df.iterrows():
            # Fetch data from Icecat API
            st.write(f"🔄 Traitement du produit {index+1}/{total} : SKU {row['sku']}")
            progress_bar.progress((index + 1) / total)
            with st.status(f"🧊 Appel API Icecat pour {row['sku']}...", expanded=False) as status_icecat:
                api_data = fetch_product_data(row)
                if not api_data:
                    st.warning(f"⚠️ Erreur Icecat pour {row['sku']}")
                    status_icecat.update(label="❌ Icecat échoué", state="error")
                    continue
                else:
                    status_icecat.update(label="✅ Icecat OK", state="complete")

            url = (
                f"https://live.icecat.biz/api?lang={languages[row['Store']]}"
                f"&Brand={row['Brand']}&ProductCode={row['PanNumber']}"
            )
            if "OpenAI content" in selected_outputs:
                with st.status("🤖 Génération de contenu OpenAI...", expanded=False) as status_openai:
                    ai_data = generate_openai_content(api_data, row, url)
                    if not ai_data:
                        st.warning(f"⚠️ OpenAI a échoué pour {row['sku']}")
                        status_openai.update(label="❌ OpenAI échoué", state="error")
                        continue
                    else:
                        status_openai.update(label="✅ OpenAI OK", state="complete")

            country = str(row['Store'])  # Convertir en chaîne
            country_dir = os.path.join(output_dir, country)
            os.makedirs(country_dir, exist_ok=True)

            if country not in writers:
                # Create files for the country
                missing_content_file = open(
                    os.path.join(country_dir, f"{today}_missing_content_{country}.csv"),
                    "w", newline='', encoding='utf-8'
                )
                status_file = open(
                    os.path.join(country_dir, f"{today}_status_{country}.csv"),
                    "w", newline='', encoding='utf-8'
                )
                price_file = open(
                    os.path.join(country_dir, f"{today}_price_{country}.csv"),
                    "w", newline='', encoding='utf-8'
                )
                processed_file = open(
                    os.path.join(country_dir, f"{today}_processed_{country}.csv"),
                    "w", newline='', encoding='utf-8'
                )
                openai_file = open(
                    os.path.join(country_dir, f"{today}_openai_content_{country}.csv"),
                    "w", newline='', encoding='utf-8'
                )

                # Create writers
                missing_content_writer = csv.DictWriter(missing_content_file, fieldnames=fieldnames_missing_content)
                status_writer = csv.DictWriter(status_file, fieldnames=fieldnames_status)
                price_writer = csv.DictWriter(price_file, fieldnames=fieldnames_price)
                processed_writer = csv.DictWriter(processed_file, fieldnames=fieldnames_processed)
                openai_writer = csv.DictWriter(openai_file, fieldnames=fieldnames_openai)

                # Write headers
                missing_content_writer.writeheader()
                status_writer.writeheader()
                price_writer.writeheader()
                processed_writer.writeheader()
                openai_writer.writeheader()

                writers[country] = {
                    "missing": missing_content_writer,
                    "status": status_writer,
                    "price": price_writer,
                    "processed": processed_writer,
                    "openai": openai_writer,
                    "files": [missing_content_file, status_file, price_file, processed_file, openai_file]
                }

            # Extract GTIN
            gtin_list = api_data["data"]["GeneralInfo"].get("GTIN", [])
            gtin = gtin_list[0] if gtin_list else ""

            # Mapping des durées de garantie vers leurs IDs
            warranty_mapping = {
                "1 year": 575,
                "2 years": 576,
                "3 years": 577,
                "4 years": 578,
                "5 years": 579,
                "6 years": 580,
                "7 years": 14639,
                "8 years": 581,
                "10 years": 582,
                "12 years": 583,
                "for life": 584,
                "3 months": 13983,
               "6 months": 13982
            }

            # Extraction de la valeur d'entrée
            warranty_value = row.get("onedirect_warranty_time", "")
            warranty_id = warranty_mapping.get(warranty_value, "")

            # Données pour contenu manquant
            missing_content_row = {
                "sku": row["sku"],
                # "onedirect_ref_fournisseur": row.get("PanNumber", ""),
                # "onedirect_ean": gtin,
                "onedirect_warranty_time": warranty_id,
                "store_id": row["Store"]
            }
            if "Missing content" in selected_outputs:
                writers[country]["missing"].writerow(missing_content_row)


            # Write status and price information
            writers[country]["status"].writerow({
                "sku": row["sku"],
                "store_id": row["Store"],
                "status": 2
            })
            writers[country]["price"].writerow({
                "sku": row["sku"],
                "website_id": row["Store"],
                "price": row.get("Price"),
                "special_price": row.get("special_price")
            })

            # Write processed file information
            processed_row = {
                "sku": row["sku"],
                "product name": row["product name"],
                "Brand": row["Brand"],
                "PanNumber": row["PanNumber"],
                "Store": row["Store"],
                "Price": row["Price"],
                "special_price": row["special_price"],
                "attribut_set": row["attribut_set"]
            }
            writers[country]["processed"].writerow(processed_row)

            # Write OpenAI content file information
            if ai_data:
                openai_row = {
                    "sku": row["sku"],
                    "name": ai_data.get("name", ""),
                    "store_id": row["Store"],
                    "onedirect_baseline": ai_data.get("baseline", ""),
                    "description": ai_data.get("description", ""),
                    "short_description": ai_data.get("features", ""),
                    "visibility": 4,
                }
                writers[country]["openai"].writerow(openai_row)

            # Consolidate product list
            sku = row["sku"]
            if sku not in unique_skus:
                images = extract_images(api_data["data"])
                consolidated_row = {
                    "sku": sku,
                    "ean": gtin,
                    "PanNumber": row["PanNumber"],
                    "Brand": row["Brand"],
                    "attribute_set_code": row["attribut_set"]
                }

                for i in range(1, 6):
                    img_field = f"img.{i}"
                    consolidated_row[img_field] = images[i - 1] if i - 1 < len(images) else ""

                consolidated_writer.writerow(consolidated_row)
                unique_skus.add(sku)

            # Finally, process the attributes for this row
            if "Attributes" in selected_outputs:
                process_attributes(row, gtin, country)
    if "Images" in selected_outputs:

        with st.status("📸 Traitement et compression des images...", expanded=False) as status_img:
            process_images(pd.read_csv(consolidated_file_path))
            status_img.update(label="✅ Images traitées et compressées", state="complete")
    # Now handle all images together from the consolidated file





def download_zip():
    """Create and download a ZIP file of the output directory."""
    # Make sure all attribute CSV files are closed
    close_attribute_writers()

    zip_path = f"{output_dir}.zip"
    shutil.make_archive(output_dir, 'zip', output_dir)

    with open(zip_path, "rb") as zip_file:
        st.download_button(
            label="Télécharger les fichiers de sortie (ZIP)",
            data=zip_file,
            file_name=f"{output_dir}.zip",
            mime="application/zip"
        )

def main():
    st.set_page_config(page_title="Traitement des produits", layout="wide")

    st.sidebar.title("🧭 Menu")
    page = st.sidebar.radio("Choisissez une page :", ["Création des imports", "Test Icecat", "Traduction en batch OpenAI"])

    if page == "Création des imports":
        page_creation_imports()
    elif page == "Test Icecat":
        page_test_icecat()
    elif page == "Traduction en batch OpenAI":
        page_openai_translation()
        
def page_creation_imports():
    st.title("📦 Création des imports")
    
    uploaded_file = st.file_uploader("📂 Chargez un fichier CSV de produits", type="csv")

    if uploaded_file is not None:
        st.success("✅ Fichier chargé avec succès !")

        # Étape 1 – Choix des fichiers à générer
        st.subheader("🗂️ Sélectionnez les fichiers à générer")
        selected_outputs = st.multiselect(
            "Que souhaitez-vous inclure dans les fichiers générés ?",
            ["Missing content", "OpenAI content", "Price", "Processed", "Status", "Attributes", "Images"],
            default=["Missing content", "OpenAI content", "Price", "Processed", "Status", "Attributes", "Images"]
        )

        # Bouton de lancement du traitement
        if st.button("🚀 Lancer le traitement"):
            process_file(uploaded_file, selected_outputs)

            for writer_info in writers.values():
                for file in writer_info["files"]:
                    file.close()

            st.success("✅ Traitement terminé. Fichiers générés avec succès.")
            download_zip()
            
def page_test_icecat():
    st.title("🧪 Test de présence des produits dans Icecat")

    uploaded_file = st.file_uploader("📂 Importez un CSV avec colonnes : sku, Brand, PanNumber, Store", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Vérification des colonnes
        required_cols = {"sku", "Brand", "PanNumber", "Store"}
        if not required_cols.issubset(df.columns):
            st.error("❌ Le fichier doit contenir les colonnes : sku, Brand, PanNumber, Store")
            return

        st.success("✅ Fichier chargé avec succès.")
        st.write(df.head())

        results = []
        progress = st.progress(0)
        total = len(df)

        icecat_key = st.secrets["ICECAT_API_TOKEN"]

        for idx, row in df.iterrows():
            sku = row["sku"]
            brand = row["Brand"]
            pan = row["PanNumber"]
            store = int(row["Store"])
            lang = languages.get(store, "en")

            url = (
                "https://live.icecat.biz/api"
                f"?UserName=Patricel"
                f"&lang={lang}"
                f"&Brand={brand}"
                f"&ProductCode={pan}"
            )

            try:
                response = requests.get(url, headers={"api-token": icecat_key}, timeout=10)
                status = response.status_code

                if status == 200:
                    comment = "Présent dans la DB Icecat"
                elif status == 403:
                    comment = "Présent mais non accessible (403)"
                elif status == 404:
                    comment = "Produit introuvable dans la DB Icecat"
                else:
                    comment = f"Erreur inconnue (code {status})"

                results.append({
                    "sku": sku,
                    "Brand": brand,
                    "PanNumber": pan,
                    "Store": store,
                    "Status": status,
                    "Commentaire": comment
                })

            except Exception as e:
                results.append({
                    "sku": sku,
                    "Brand": brand,
                    "PanNumber": pan,
                    "Store": store,
                    "Status": "Error",
                    "Commentaire": f"Erreur lors de la requête : {e}"
                })

            progress.progress((idx + 1) / total)

        result_df = pd.DataFrame(results)
        st.success("✅ Vérification terminée.")
        st.dataframe(result_df)

        # Téléchargement
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger le rapport CSV",
            data=csv,
            file_name="rapport_test_icecat.csv",
            mime="text/csv"
        )
        
def page_openai_translation():
    st.title("🌍 Traduction multicolonne avec OpenAI")

    uploaded_file = st.file_uploader("📂 Chargez un fichier CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Fichier chargé avec succès")
        st.write(df.head())

        # Sélection des colonnes à traduire
        selectable_columns = list(df.columns)
        selected_columns = st.multiselect("📝 Sélectionnez les colonnes à traduire :", selectable_columns)

        if not selected_columns:
            st.warning("⚠️ Veuillez sélectionner au moins une colonne.")
            return

        # Sélection de la langue cible
        selected_countries = st.multiselect("🌐 Choisissez le pays cible :", ["FR", "UK", "ES", "PT", "IT", "DE", "NL"])
        lang_map = {
            "FR": "Vous êtes un traducteur professionnel, de langue maternelle française, spécialisé dans les contenus de télécommunications. Vous traduisez pour Onedirect, une entreprise spécialisée dans les solutions professionnelles de communication et de téléphonie. Votre objectif est de fournir une traduction :\n\n- 100  fidèle au sens et à la terminologie d'origine,\n- Fluide et naturelle pour un lecteur francophone natif,\n- Fidèle au ton et à la structure du texte original,\n- Adaptée à une utilisation commerciale et professionnelle.\n\nVeuillez traduire le texte suivant en français naturel et idiomatique. Conservez le formatage (titres, puces, gras, etc.) et n'ajoutez aucun commentaire.\n\nRetournez uniquement le texte traduit en français.",
            "UK": "You are a professional translator and native English speaker specializing in telecommunications content. You are translating on behalf of Onedirect, a company that sells business communication and telephony solutions. Your goal is to provide a translation that is:\n\n- 100 accurate in meaning and terminology,\n- Fluent and natural for native English readers,\n- Faithful to the tone and structure of the original,\n- Suitable for commercial and professional use.\n\nPlease translate the following text into natural, idiomatic English. Keep formatting (headings, bullets, bold, etc.) and do not add any commentary.\n\nReturn only the translated English text.",
            "ES": "Eres un traductor profesional, hispanohablante nativo, especializado en contenidos sobre telecomunicaciones. Traduces para Onedirect, una empresa que ofrece soluciones profesionales en comunicación empresarial y telefonía. Tu objetivo es proporcionar una traducción que sea:\n\n- 100 % precisa en cuanto a significado y terminología,\n- Fluida y natural para lectores hispanohablantes nativos,\n- Fiel al tono y estructura del texto original,\n- Adecuada para un uso comercial y profesional.\n\nPor favor, traduce el siguiente texto a un español natural e idiomático. Mantén el formato original (títulos, viñetas, negrita, etc.) y no añadas ningún comentario.\n\nDevuelve únicamente el texto traducido en español, siguiendo estrictamente las normas y el vocabulario del español europeo.",
            "PT": "És um tradutor profissional, falante nativo de português europeu, especializado em conteúdos de telecomunicações. Estás a traduzir em nome da Onedirect, uma empresa que vende soluções empresariais de comunicação e telefonia. O teu objetivo é fornecer uma tradução:\n\n- 100% precisa no significado e na terminologia,\n- Fluida e natural para leitores nativos de português europeu,\n- Fiel ao tom e à estrutura do texto original,\n- Adequada para uso comercial e profissional.\n\nPor favor, traduz o seguinte texto para um português natural e idiomático. Mantém a formatação original (títulos, pontos, negritos, etc.) e não acrescentes comentários.\n\nDevolve apenas o texto traduzido em português, seguindo rigorosamente as regras e o vocabulário do português europeu.",
            "IT": "Sei un traduttore professionista, madrelingua italiano, specializzato in contenuti relativi alle telecomunicazioni. Traduci per conto di Onedirect, un'azienda che vende soluzioni professionali per comunicazione aziendale e telefonia. Il tuo obiettivo è fornire una traduzione:\n\n- Precisa al 100% nel significato e nella terminologia,\n- Fluida e naturale per lettori madrelingua italiani,\n- Fedele al tono e alla struttura del testo originale,\n- Adatta per uso commerciale e professionale.\n\nTraduci il seguente testo in italiano naturale e idiomatico. Conserva la formattazione originale (titoli, elenchi puntati, grassetto, ecc.) e non aggiungere alcun commento.\n\nRestituisci solo il testo tradotto in italiano.",
            "DE": "Du bist ein professioneller Übersetzer und deutscher Muttersprachler mit Spezialisierung auf Telekommunikationsinhalte. Du übersetzt im Auftrag von Onedirect, einem Unternehmen, das professionelle Lösungen für Geschäftskommunikation und Telefonie anbietet. Dein Ziel ist es, eine Übersetzung zu liefern, die:\n\n- Zu 100  genau in Bedeutung und Terminologie ist,\n- Flüssig und natürlich für deutschsprachige Muttersprachler klingt,\n- Dem Ton und der Struktur des Originaltexts treu bleibt,\n- Für kommerzielle und professionelle Nutzung geeignet ist.\n\nBitte übersetze den folgenden Text in natürliches, idiomatisches Deutsch. Behalte die Originalformatierung (Überschriften, Aufzählungen, Fettdruck usw.) bei und füge keine Kommentare hinzu.\n\nGib ausschließlich den übersetzten deutschen Text zurück.",
            "NL": "Je bent een professionele vertaler en moedertaalspreker Nederlands, gespecialiseerd in telecomgerelateerde inhoud. Je vertaalt namens Onedirect, een bedrijf dat professionele communicatie- en telefonieoplossingen voor bedrijven verkoopt. Jouw doel is een vertaling te leveren die:\n\n- 100% nauwkeurig is qua betekenis en terminologie,\n- Vloeiend en natuurlijk klinkt voor Nederlandstalige lezers,\n- De toon en structuur van de originele tekst trouw volgt,\n- Geschikt is voor commercieel en professioneel gebruik.\n\nVertaal de volgende tekst naar natuurlijk en idiomatisch Nederlands. Behoud de originele opmaak (koppen, opsommingstekens, vetgedrukt, enz.) en voeg geen commentaar toe.\n\nStuur alleen de vertaalde Nederlandse tekst terug."
        }

        

        if st.button("🚀 Lancer la traduction"):
            st.info(f"🧠 Traduction en cours")
            openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
            if not openai_key:
                st.error("❌ Clé API OpenAI manquante")
                return

            client = OpenAI(api_key=openai_key)
            total_tasks = len(df) * (len(selected_columns) * len(selected_countries))
            progress = st.progress(0)
            task_count = 0
            for country in selected_countries :
                for col in selected_columns:
                    target_language = lang_map.get(country, "English")
                    translated_col = f"{col}_translated_{country}"
                    df[translated_col] = ""

                    for i in range(len(df)):
                        original_text = str(df.at[i, col])

                        prompt = f"{target_language}:\n{original_text} \n\n If bold use <strong>"
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                temperature=0.2,
                                max_tokens=2048,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            translated_text = response.choices[0].message.content.strip()
                        except Exception as e:
                            translated_text = f"[Erreur OpenAI] {e}"

                        df.at[i, translated_col] = translated_text
                        task_count += 1
                        progress.progress(task_count / total_tasks)

            st.success("✅ Traduction terminée")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger le fichier traduit",
                data=csv,
                file_name=f"traduction_multicolonne_{country}.csv",
                mime="text/csv"
            )



if __name__ == "__main__":
    main()
