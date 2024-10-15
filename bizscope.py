import streamlit as st
import pandas as pd
import requests
import aiohttp
import asyncio
import aiofiles
from io import StringIO
import matplotlib.pyplot as plt
from rapidfuzz import process, fuzz
from aiohttp_retry import RetryClient, ExponentialRetry


def process_uploaded_file(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8')
    company_blocks = content.strip().split("\n\n")
    companies = [parse_company_data(block) for block in company_blocks]
    return pd.DataFrame(companies)


def parse_company_data(block):
    lines = block.strip().split("\n")
    company = {
        'name': lines[0] if len(lines) > 0 else "Unknown",
        'employees': lines[1] if len(lines) > 1 else "Unknown",
        'category': lines[2] if len(lines) > 2 else "Unknown",
        'location': lines[3] if len(lines) > 3 else "Unknown",
        'id': lines[4] if len(lines) > 4 else "Unknown",
        'status': lines[-2] if len(lines) > 6 else "Unknown",
        'contact': lines[-1] if '@' in lines[-1] else "Unknown"
    }
    return company


async def fetch_company_data(session, company_name, api_key, rate_limit_delay=1):
    base_url = f'https://api.opencorporates.com/v0.4/companies/{company_name}'
    params = {'api_token': api_key}

    try:
        await asyncio.sleep(rate_limit_delay)
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data and 'company' in data['results']:
                    company_info = data['results']['company']
                    return {
                        'company_number': company_info.get('company_number', 'Unknown'),
                        'status': company_info.get('current_status', 'Unknown')
                    }
            else:
                st.warning(f"Failed to fetch {company_name}: HTTP {response.status}")
    except aiohttp.ClientError as e:
        st.error(f"Error fetching data for {company_name}: {str(e)}")
    return {}


async def enrich_company_data_async(companies_df, api_key, rate_limit_delay=1):
    retry_options = ExponentialRetry(attempts=3)
    async with RetryClient(raise_for_status=False, retry_options=retry_options) as session:
        tasks = []
        for index, row in companies_df.iterrows():
            tasks.append(fetch_company_data(session, row['name'], api_key, rate_limit_delay))
        enriched_data = await asyncio.gather(*tasks)
        for i, data in enumerate(enriched_data):
            companies_df.at[i, 'registration_number'] = data.get('company_number', 'Unknown')
            companies_df.at[i, 'status'] = data.get('status', companies_df.at[i, 'status'])

    return companies_df


def enrich_company_data(companies_df, api_key, rate_limit_delay=1):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(enrich_company_data_async(companies_df, api_key, rate_limit_delay))


def main():
    st.title("OSINT Company Research Tool")

    with st.sidebar:
        st.subheader("Upload and Filter Data")
        uploaded_file = st.file_uploader("Upload company data file", type="txt")
        name_filter = st.text_input("Company Name (fuzzy search)")
        location_filter = st.text_input("Location")

    if uploaded_file is not None:
        st.subheader("Company Data Preview")
        companies_df = process_uploaded_file(uploaded_file)
        st.write(companies_df.head())
        filtered_df = apply_filters(companies_df, name_filter, location_filter)
        st.write(f"Filtered Results: {len(filtered_df)} companies found.")
        st.write(filtered_df)

        if st.button("Enrich Data"):
            api_key = st.text_input("OpenCorporates API Key (required for enrichment)", type="password")
            if api_key:
                rate_limit_delay = st.slider("Rate Limit Delay (seconds)", min_value=1, max_value=10, value=1)
                enriched_df = enrich_company_data(filtered_df, api_key, rate_limit_delay)
                st.write("Enriched Data:")
                st.write(enriched_df)
            else:
                st.error("Please provide an API key for data enrichment.")

        if not filtered_df.empty:
            st.subheader("Download Data")
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download Filtered Data as CSV", data=csv, mime="text/csv")

        with st.expander("Company Distribution by Location"):
            st.subheader("Company Distribution by Location")
            location_counts = filtered_df['location'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(location_counts.index, location_counts.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)


def apply_filters(companies_df, name_filter, location_filter):
    filtered_df = companies_df
    if name_filter:
        filtered_df['name_match'] = filtered_df['name'].apply(lambda x: fuzz.partial_ratio(name_filter, x))
        filtered_df = filtered_df[filtered_df['name_match'] > 80]
    if location_filter:
        filtered_df = filtered_df[filtered_df['location'].str.contains(location_filter, case=False, na=False)]

    return filtered_df


if __name__ == '__main__':
    main()
