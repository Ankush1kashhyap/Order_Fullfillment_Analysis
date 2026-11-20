# 📦 Order Fulfillment EDA Dashboard

An **interactive Streamlit analytics dashboard** for India E-Commerce Order Fulfillment data — covering KPIs, carrier performance, warehouse bottlenecks, delivery quality, cancellations, and statistical hypothesis testing.

---

## 🚀 Live Demo

Run locally at: `http://localhost:8501`

---

## 📊 Features

| Page | Description |
|------|-------------|
| 🏠 **Overview** | 7 KPIs, 4 animated gauges, monthly trend, revenue treemap |
| 📈 **Order Trends** | Time series, MoM growth, product mix sunburst, calendar heatmap |
| 🚛 **Carrier Analysis** | Carrier comparator tool, violin plots, cost vs speed scatter |
| 🏭 **Warehouse Deep-Dive** | O2S bottleneck analysis + Welch's t-test significance |
| 🎯 **Quality & Returns** | Delivery condition stacked bars, OTIF breakdown |
| ❌ **Cancellations** | Pareto chart, revenue lost, region & segment breakdown |
| 🔬 **Statistical Tests** | Live hypothesis testing, correlation heatmap, scatter matrix |
| 📊 **Executive Report** | Priority findings, recommendations, CSV download |

## 🎛️ Interactive Controls
- **5 Global Sidebar Filters**: Date range, Region, Segment, Shipping Tier, Category, Warehouse
- All charts react live to filter changes
- Carrier-to-carrier comparison tool
- Per-warehouse drill-down
- Distribution explorer (any variable × any grouping)

---

## 🛠️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/order-fulfillment-eda.git
cd order-fulfillment-eda
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python generate_dataset.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🚀

---

## 📁 Project Structure

```
order-fulfillment-eda/
│
├── app.py                              # 🎯 Main Streamlit dashboard
├── generate_dataset.py                 # 🔧 Synthetic dataset generator
├── requirements.txt                    # 📦 Python dependencies
├── dashboard.html                      # 🌐 Static HTML dashboard (Chart.js)
├── order_fulfillment_eda_dataset.csv   # 📊 Generated dataset (562 orders)
├── Order_Fulfillment_EDA_Report.html   # 📄 EDA report
└── README.md
```

---

## 📈 Dataset Overview

| Field | Description |
|-------|-------------|
| `order_id` | Unique order identifier |
| `order_date` | Order placement timestamp |
| `warehouse_id/name` | Fulfillment center (6 warehouses across India) |
| `dest_region/state/city` | Delivery destination |
| `customer_segment` | VIP / Retail / SMB |
| `product_category` | 6 categories (Books, Apparel, Electronics, etc.) |
| `shipping_tier` | Same-Day / Express / Standard |
| `carrier_name` | 5 carriers (EcoParcel, PrimeFreight, etc.) |
| `order_status` | Delivered / Cancelled / Failed |
| `order_to_ship_hours` | Pick-pack speed at warehouse |
| `transit_hours` | Carrier delivery time |
| `on_time_flag` | 1 = met SLA, 0 = missed |
| `in_full_flag` | 1 = full qty delivered |
| `otif_flag` | On-Time In-Full (computed) |
| `delivery_condition` | Intact / Damaged / Returned |

---

## 🔍 Key Findings

1. **🔴 WH-04 Kolkata Bottleneck** — 41.7h O2S vs 25h network avg (p < 0.0001)
2. **🔴 EcoParcel** — 17.5% on-time vs 50.5% for PrimeFreight
3. **🔴 OTIF 37.2%** — far below 85% SLA target (systemic failure)
4. **🟡 Stockout** — #1 cancellation reason driving 15.1% cancel rate
5. **🟡 Oct/Nov surge** — festive peaks strain capacity every year
6. **🟢 Apparel** — highest damage rate (13.8%) despite being non-fragile

---

## 🧰 Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web app framework
- **[Plotly](https://plotly.com/)** — Interactive charts
- **[Pandas](https://pandas.pydata.org/)** — Data manipulation
- **[SciPy](https://scipy.org/)** — Statistical testing
- **[streamlit-option-menu](https://github.com/victoryhb/streamlit-option-menu)** — Sidebar navigation

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
