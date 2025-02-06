import streamlit as st
import pandas as pd
import os
import requests
import csv
import json
import shutil
from datetime import date
from urllib.parse import urlparse
from openai import OpenAI
from PIL import Image, ImageOps
from io import BytesIO
import zipfile
from dotenv import load_dotenv


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
fieldnames_missing_content = ["sku", "onedirect_ref_fournisseur", "onedirect_ean", "onedirect_warranty_time", "store_id"]
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

def clean_openai_response(content):
    """Clean the OpenAI response by removing code block markers."""
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


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
    """Generate content using OpenAI API."""
    ai_prompt = """
Here is an example of an URL for a product on the OneDirect website: {0}
You are a writer for the {3} website that specializes in tech products and telecommunications devices. I need you to write a name (I want a short name), a baseline, a description divided in multiple paragrph (max 500 words/eatch paragrph MUST strat with an h3)
with HTML markings and the key features (max 7 items / those point MUST be short and impact full kind as markting catchline) also with HTML markings for a new product whose information is available in the following JSON: {1}.
The product name is {2}. You should use the same style as in the provided example URL.
The baseline MUST be include between <h2> html tags.
The description MUST be include between <p> html tags.
The features MUST be include between <ul> html tags.
Escaped quotes in the content are ESSENTIAL for valid JSON (/\").
I need you to extract all the weights and dimensions of the package for this product and include them in the result JSON. If you don't find a dimension or the weight of the package in the info, leave the corresponding JSON field blank.
Return a JSON with the following fields: baseline, description, features, weight, depth, height, and width.
Return a JSON with the following fields: name, baseline, description, features, weight, depth, height, and width.
If you encounter any problem, return an empty JSON.
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
                    "content": ai_prompt.format(url, infos, title, urlparse(url).hostname)
                }
            ]
        )

        # Debug: Log the raw response
        # st.write("Raw OpenAI Response:", response)

        # Attempt to parse the response
        if response.choices and response.choices[0].message.content:
            cleaned_content = clean_openai_response(response.choices[0].message.content)
            return json.loads(cleaned_content)
        else:
            st.error("Réponse vide ou invalide reçue de l'API OpenAI.")
            return None

    except json.JSONDecodeError as json_err:
        st.error(f"Erreur de décode JSON : {str(json_err)}")
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


def process_images(df):
    """Télécharge, traite et compresse les images dans un fichier ZIP."""
    with zipfile.ZipFile(images_zip_path, 'w') as zipf:
        for _, row in df.iterrows():
            sku = row.get("sku")
            additional_images = []
            base_image = small_image = thumbnail_image = ""

            for i in range(1, 31):
                img_url = row.get(f"img.{i}")

                if pd.notna(img_url) and img_url:
                    try:
                        # Télécharge l'image
                        response = requests.get(img_url, timeout=10)
                        if response.status_code == 200:
                            image = Image.open(BytesIO(response.content))

                            # Rend l'image carrée (800x800)
                            image_square = ImageOps.pad(image, (800, 800), color="white", centering=(0.5, 0.5))

                            # Nomme l'image en fonction du SKU et de l'index
                            img_filename = f"{sku}_{i}.jpg"

                            # Ajoute l'image au fichier ZIP
                            with BytesIO() as img_buffer:
                                image_square.save(img_buffer, format="JPEG")
                                zipf.writestr(img_filename, img_buffer.getvalue())

                            # Assigne les noms des images aux colonnes correspondantes
                            if i == 1:
                                base_image = img_filename
                                small_image = img_filename
                                thumbnail_image = img_filename
                            else:
                                additional_images.append(img_filename)
                    except Exception as e:
                        st.error(f"Erreur lors du traitement de l'image {img_url}: {e}")

            # Ajoute les informations d'image au CSV
            image_data.append({
                "sku": sku,
                "base_image": base_image,
                "small_image": small_image,
                "thumbnail_image": thumbnail_image,
                "additional_images": ','.join(additional_images)
            })

    # Génère un CSV avec les chemins d'images
    pd.DataFrame(image_data).to_csv(images_csv_path, index=False)
    
    
def process_file(file):
    """Process the uploaded CSV file and generate outputs."""
    df = pd.read_csv(file)
    st.write("Fichier chargé :")
    st.write(df.head())

    # Open consolidated file
    with open(consolidated_file_path, "w", newline='', encoding='utf-8') as consolidated_file:
        consolidated_writer = csv.DictWriter(consolidated_file, fieldnames=fieldnames_consolidated)
        consolidated_writer.writeheader()

        for index, row in df.iterrows():
            # Fetch data from Icecat API
            api_data = fetch_product_data(row)
            if not api_data:
                continue

            url = (
                f"https://live.icecat.biz/api?lang={languages[row['Store']]}"
                f"&Brand={row['Brand']}&ProductCode={row['PanNumber']}"
            )
            ai_data = generate_openai_content(api_data, row, url)
            if not ai_data:
                continue

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

            # Data for missing content
            missing_content_row = {
                "sku": row["sku"],
                "onedirect_ref_fournisseur": row.get("PanNumber", ""),
                "onedirect_ean": gtin,
                "onedirect_warranty_time": row.get("onedirect_warranty_time", ""),
                "store_id": row["Store"]
            }
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

                for i in range(1, 31):
                    img_field = f"img.{i}"
                    consolidated_row[img_field] = images[i - 1] if i - 1 < len(images) else ""

                consolidated_writer.writerow(consolidated_row)
                unique_skus.add(sku)

            # Finally, process the attributes for this row
            process_attributes(row, gtin, country)

    # Now handle all images together from the consolidated file
    process_images(pd.read_csv(consolidated_file_path))




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
    st.title("Traitement avancé des produits avec Streamlit")

    uploaded_file = st.file_uploader("Chargez un fichier CSV", type="csv")

    if uploaded_file is not None:
        process_file(uploaded_file)

        # Close all "writers" that were opened
        for writer_info in writers.values():
            for file in writer_info["files"]:
                file.close()

        st.success("Fichiers créés avec succès dans le dossier de sortie.")

        # Allow download of ZIP file
        download_zip()


if __name__ == "__main__":
    main()
