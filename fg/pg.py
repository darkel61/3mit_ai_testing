import psycopg2
import pandas as pd

DBs = ['fg_v1', 'fg_v2']

def connect_db(db_name):
    pg_config = {
        "user": "fernandoazuaje",
        "password": "",
        "host": "localhost",
        "port": "5432",
        "database": db_name
    }

    conn = psycopg2.connect(**pg_config)
    return conn

sql_query = """
select 
	pt.principio_activo,
	pt.name,
	date(pol.create_date),
	pol.qty,
	pol.price_unit,
    po.branch_id
from pos_order_line pol 
left join pos_order po on pol.order_id = po.id
left join product_product pp on pol.product_id = pp.id 
left join product_template pt on pp.product_tmpl_id = pt.id 
where 
po.state = 'invoiced' and
pt.principio_activo in ('DICLOFENAC POTASICO', 'ACETAMINOFEN');
"""

branch_by_day_query = """
select 
    date(po.create_date),
    count(DISTINCT po.branch_id)
from pos_order as po
where po.state = 'invoiced'
group by 
date(po.create_date)
"""

products_by_day_query = """
select 
	date(po.create_date),
	pt.principio_activo ,
	count(distinct pt.id)
from pos_order_line pol 
left join pos_order po on pol.order_id = po.id
left join product_product pp on pol.product_id = pp.id 
left join product_template pt on pp.product_tmpl_id = pt.id 
where pt.principio_activo in ('DICLOFENAC POTASICO', 'ACETAMINOFEN')
and po.state = 'invoiced' 
group by pt.principio_activo, date(po.create_date)
order by date(po.create_date)
"""

def load_data(query, file_name):

    list_df = []
    for db_name in DBs:
        df = pd.read_sql_query(query, connect_db(db_name))
        list_df.append(df)
    df_full = pd.concat(list_df, axis=0, ignore_index=True)

    df_full.to_csv(f"./fg/{file_name}.csv", index=False)

    return df_full
            
## 
load_data(sql_query, 'ts_fg')
load_data(branch_by_day_query, 'branch_by_day')
load_data(products_by_day_query, 'products_by_day')

