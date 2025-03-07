#!D:/burundi/.venv/Scripts/python.exe
 
# Block 1: Import packages 
import os 
import json 
import psycopg2 
from psycopg2.extras import RealDictCursor 
import cgi
 
# Block 2 : Get inputs from the client
form = cgi.FieldStorage() 
centroid = form.getvalue("location")
centroid = centroid.split(",")
radious = form.getvalue("distance")
srid = form.getvalue("srid")
 
# Block 3: connect to the database 
file = open(os.path.dirname(os.path.abspath(__file__)) + "\db.credentials") 
connection_string = file.readline() + file.readline() 
pg_conn = psycopg2.connect(connection_string) 
pg_cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
 
# Block 4: Query data
selectQuery = """
    SELECT a.name, a.categorie, ST_asGEOJson(ST_Transform(a.geom, %d)) as geom
    FROM vector.bi_markets AS a
    WHERE st_dwithin(ST_Transform(a.geom, %d), ST_GeomFromText('POINT(%s %s)', %d), %s)
""" %(int(srid), int(srid), float(centroid[0]), float(centroid[1]), int(srid), float(radious) * 1000.0)
 
pg_cursor.execute(selectQuery) 
records = pg_cursor.fetchall()
markets = json.dumps(records)
pg_conn.close()
 
# Block 5: Return result to the client 
print("Content-type: application/json")
print()
print(markets)