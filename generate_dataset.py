import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

n = 556
warehouses = [('WH-01','Delhi North Hub'),('WH-02','Gurugram North DC'),('WH-03','Bengaluru South FC'),
              ('WH-04','Kolkata East Hub'),('WH-05','Hyderabad Central DC'),('WH-06','Jaipur Express Node')]
regions_states = {
    'North':['DL','UP','HR','PB','RJ'],
    'South':['TN','KA','AP','KL'],
    'East':['WB','OR','JH','AS'],
    'West':['MH','GJ'],
    'Central':['CG','MP']
}
cities = {
    'DL':['Delhi'],'UP':['Lucknow','Kanpur'],'HR':['Gurgaon'],'PB':['Amritsar'],'RJ':['Jaipur','Jodhpur'],
    'TN':['Chennai'],'KA':['Bengaluru'],'AP':['Hyderabad'],'KL':['Kochi'],
    'WB':['Kolkata'],'OR':['Bhubaneswar'],'JH':['Ranchi'],'AS':['Guwahati'],
    'MH':['Mumbai','Pune'],'GJ':['Ahmedabad'],'MP':['Bhopal'],'CG':['Raipur']
}
categories = ['Books & Media','Apparel','Electronics','Sports & Fitness','Home & Kitchen','Health & Beauty']
segments = ['VIP','Retail','SMB']
tiers = {'Same-Day':24,'Express':48,'Standard':120}
# Correct carrier names the notebook expects
carriers = ['EcoParcel','PrimeFreight','SwiftMove','RapidEx','CargoKing']
n_delivered = 469
n_cancelled = 80
n_failed = 7
cancel_reasons_list = ['Stockout','Customer Request','Address Unreachable','Payment Failure','Duplicate Order']
conditions_pool = ['Intact']*380 + ['Damaged']*50 + ['Returned']*39
random.shuffle(conditions_pool)

rows = []
statuses = ['Delivered']*n_delivered + ['Cancelled']*n_cancelled + ['Failed']*n_failed
random.shuffle(statuses)

cond_idx = 0
for i in range(n):
    wh = random.choice(warehouses)
    region = random.choice(list(regions_states.keys()))
    state = random.choice(regions_states[region])
    city_list = cities.get(state, ['Unknown'])
    city = random.choice(city_list)
    seg = random.choice(segments)
    cat = random.choice(categories)
    fragile = 1 if cat == 'Electronics' or random.random() < 0.2 else 0
    tier_name = random.choice(list(tiers.keys()))
    sla_hours = tiers[tier_name]
    status = statuses[i]

    if status == 'Delivered':
        carrier = random.choice(carriers)
        # EcoParcel has wider/slower transit, PrimeFreight is fast and consistent
        if carrier == 'EcoParcel':
            o2s = round(random.uniform(10, 55), 2)
            transit = round(random.uniform(35, 106), 2)
        elif carrier == 'PrimeFreight':
            o2s = round(random.uniform(5, 25), 2)
            transit = round(random.uniform(12, 45), 2)
        else:
            o2s = round(random.uniform(5, 50), 2)
            transit = round(random.uniform(12, 80), 2)
        # WH-04 Kolkata East Hub - much slower O2S
        if wh[0] == 'WH-04':
            o2s = round(random.uniform(30, 55), 2)
        total = round(o2s + transit, 2)
        on_time = 1 if total <= sla_hours else 0
        cancel_reason = None
        ship_cost = round(random.uniform(5, 50), 2)
        cond = conditions_pool[cond_idx % len(conditions_pool)]
        cond_idx += 1
    else:
        carrier = None
        o2s = None
        transit = None
        total = None
        on_time = 0
        cancel_reason = random.choice(cancel_reasons_list)
        ship_cost = 0.0
        cond = 'Unfulfilled'

    qty = random.randint(1, 6)
    del_qty = qty if status == 'Delivered' else 0
    in_full = 1 if del_qty == qty else 0
    weight = round(random.uniform(0.3, 14), 2)
    value = round(random.uniform(19, 1538), 2)

    base_date = datetime(2024, 7, 1) + timedelta(days=random.randint(0, 540))
    rows.append([
        f'ORD-2025-{i+1:04d}',
        base_date.strftime('%Y-%m-%d %H:%M:%S'),
        wh[0], wh[1], region, state, city, seg, cat, fragile, tier_name, sla_hours,
        carrier, status, cancel_reason, qty, del_qty, weight, value, ship_cost,
        o2s, transit, total, on_time, in_full, cond
    ])

# Patch first 2 rows to match the notebook's expected head() output
rows[0] = ['ORD-2025-0001','2024-11-19 14:15:00','WH-02','Gurugram North DC','South','TN','Chennai',
           'VIP','Books & Media',0,'Standard',120,None,'Cancelled','Stockout',4,0,3.89,613.32,0.0,
           None,None,None,0,0,'Unfulfilled']
rows[1] = ['ORD-2025-0002','2024-07-16 12:15:00','WH-03','Bengaluru South FC','Central','CG','Raipur',
           'Retail','Apparel',0,'Express',48,None,'Cancelled','Customer Request',6,0,2.82,447.66,0.0,
           None,None,None,0,0,'Unfulfilled']

cols = ['order_id','order_date','warehouse_id','warehouse_name','dest_region','dest_state','dest_city',
        'customer_segment','product_category','is_fragile','shipping_tier','sla_target_hours',
        'carrier_name','order_status','cancellation_reason','ordered_qty','delivered_qty',
        'order_weight_kg','order_value_usd','shipping_cost_usd','order_to_ship_hours',
        'transit_hours','total_fulfillment_hours','on_time_flag','in_full_flag','delivery_condition']

df = pd.DataFrame(rows, columns=cols)

# Add 6 duplicate order_id rows (notebook Section 3 expects exactly 6 duplicates)
dup_rows = df.sample(6, random_state=1).copy()
df = pd.concat([df, dup_rows], ignore_index=True)

# Introduce unexpected missing values on delivered rows (notebook expects 6 each)
delivered_idx = df[df['order_status'] == 'Delivered'].index.tolist()
for idx in delivered_idx[:6]:
    df.at[idx, 'dest_city'] = None
for idx in delivered_idx[6:12]:
    df.at[idx, 'carrier_name'] = None
for idx in delivered_idx[12:18]:
    df.at[idx, 'shipping_cost_usd'] = None

# Add known bad labels (notebook Section 3 expects these)
elec_idx = df[df['product_category'] == 'Electronics'].index[0]
df.at[elec_idx, 'product_category'] = 'electronics '
exp_idx = df[df['shipping_tier'] == 'Express'].index[0]
df.at[exp_idx, 'shipping_tier'] = 'express'

df.to_csv('order_fulfillment_eda_dataset.csv', index=False)

print(f"CSV created: {len(df)} rows")
delivered = df[df['order_status'] == 'Delivered']
print(f"Delivered: {len(delivered)}, Cancelled+Failed: {len(df) - len(delivered)}")
print(f"EcoParcel count: {(df['carrier_name'] == 'EcoParcel').sum()}")
print(f"PrimeFreight count: {(df['carrier_name'] == 'PrimeFreight').sum()}")
print(f"Carriers: {df['carrier_name'].dropna().unique().tolist()}")
print(f"Statuses: {df['order_status'].value_counts().to_dict()}")
