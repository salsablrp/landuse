#!D:/burundi/.venv/Scripts/python.exe

# Block 1: Import packages 
import os 
import json 
import psycopg2 
from psycopg2.extras import RealDictCursor 
import cgi

# Block 2 : Get inputs from the client
form = cgi.FieldStorage() 
coord = form.getvalue("location") 
coord = coord.split(",") 
srid = form.getvalue("srid")

# Block 3: connect to the database 
file = open(os.path.dirname(os.path.abspath(__file__)) + "\db.credentials") 
connection_string = file.readline() + file.readline() 
pg_conn = psycopg2.connect(connection_string) 
pg_cursor = pg_conn.cursor(cursor_factory=RealDictCursor)

# Block 4: Query data
selectQuery = """
    SELECT a.gid, b.pop_2020::integer, a.name, b.categorie, st_asgeojson(st_transform(b.geom, 3857)) as geometry
        FROM vector.bi_small_areas as a JOIN vector.bi_markets as b 
        ON st_intersects(a.geom, b.geom) AND b.categorie = 'small_markets' 
        WHERE st_intersects(a.geom, ST_Transform(ST_GeomFromText('POINT(%s %s)', %d), st_srid(a.geom)))
    union
    SELECT distinct a.gid, b.pop_2020::integer, a.name, b.categorie, st_asgeojson(st_transform(b.geom, 3857))  as geometry
        FROM vector.bi_medium_areas as a JOIN vector.bi_markets as b 
        ON st_intersects(a.geom, b.geom) AND b.categorie = 'medium_markets' 
        WHERE st_intersects(a.geom, ST_Transform(ST_GeomFromText('POINT(%s %s)', %d), st_srid(a.geom)))
    union
    SELECT a.gid, b.pop_2020::integer, a.name, b.categorie, st_asgeojson(st_transform(b.geom, 3857))  as geometry
        FROM vector.bi_local_areas as a JOIN vector.bi_markets as b 
        ON st_intersects(a.geom, b.geom) AND b.categorie = 'local_markets' 
        WHERE st_intersects(a.geom, ST_Transform(ST_GeomFromText('POINT(%s %s)', %d), st_srid(a.geom)))
    union
    SELECT a.gid, b.pop_2020::integer, a.name, b.categorie, st_asgeojson(st_transform(b.geom, 3857))  as geometry
        FROM vector.bi_capital_areas as a JOIN vector.bi_markets as b 
        ON st_intersects(a.geom, b.geom) AND b.categorie = 'capital_markets' 
        WHERE st_intersects(a.geom, ST_Transform(ST_GeomFromText('POINT(%s %s)', %d), st_srid(a.geom)))
""" % (float(coord[0]), float(coord[1]), int(srid), 
        float(coord[0]), float(coord[1]), int(srid), 
        float(coord[0]), float(coord[1]), int(srid), 
        float(coord[0]), float(coord[1]), int(srid))

pg_cursor.execute(selectQuery) 
records = pg_cursor.fetchall()
markets = json.dumps(records)
pg_conn.close()

# Block 5: Return result to the client 
print("Content-type: application/json") 
print() 
print(markets)

