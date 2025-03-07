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
size = form.getvalue("size")
srid = form.getvalue("srid")
 
# Block 3: connect to the database 
file = open(os.path.dirname(os.path.abspath(__file__)) + "\db.credentials") 
connection_string = file.readline() + file.readline() 
pg_conn = psycopg2.connect(connection_string) 
pg_cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
 
# Block 4: Query data
selectQuery = """
    SELECT name, st_asgeojson(st_transform(geom, %d)) as geom
    FROM vector.bi_%s AS a
    WHERE st_intersects(a.geom, st_transform(st_geomfromtext('POINT(%s %s)', %d), st_srid(a.geom)))
""" % (int(srid), size, float(coord[0]), float(coord[1]), int(srid))
 
pg_cursor.execute(selectQuery) 
records = pg_cursor.fetchall() 
market_area = json.dumps(records)
pg_conn.close()
 
# Block 5: Return result to the client 
print("Content-type: application/json") 
print() 
print(market_area)