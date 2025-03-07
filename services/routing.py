#!D:/burundi/.venv/Scripts/python.exe
 
# Block 1: Import packages 
import os 
import json 
import psycopg2 
from psycopg2.extras import RealDictCursor 
import cgi
 
# Block 2 : Get inputs from the client
form = cgi.FieldStorage() 
source = form.getvalue('source') 
target = form.getvalue('target') 
srid = form.getvalue('srid')
source = source.split(",") 
target = target.split(",")
srid = int(srid)
 
# Block 3: connect to the database 
file = open(os.path.dirname(os.path.abspath(__file__)) + "\db.credentials") 
connection_string = file.readline() + file.readline() 
pg_conn = psycopg2.connect(connection_string) 
pg_cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
 
# Block 4: Define a function to get the closest node on the network
def get_closest_node(coord, srid): 
    selectQuery = """SELECT id, 
        ST_Distance(the_geom, ST_Transform(ST_GeomFromText('POINT(%s %s)', %d), st_srid(the_geom))) AS distance
        FROM vector.bi_main_roads_vertices_pgr  
        ORDER BY 2 ASC
        LIMIT 1""" % (float(coord[0]), float(coord[1]), srid) 
    pg_cursor.execute(selectQuery) 
    result = pg_cursor.fetchall() 
    if len(result) > 0: 
        return result[0]
 
def get_shortest_path(source_coord, target_coord, srid): 
    origin_node = get_closest_node(source_coord, srid) 
    target_node = get_closest_node(target_coord, srid)
    if origin_node is None or target_node is None: 
        return None 
    origin_id = origin_node["id"] 
    target_id = target_node["id"]
    selectQuery = """
        SELECT st_asgeojson(st_transform(st_union(geom), %d))::json as path, sum(st_length(geom)) as distance FROM vector.bi_main_roads
        WHERE gid IN ( 
        SELECT edge  
        FROM pgr_dijkstra( 
        'SELECT gid id, source, target, cost FROM vector.bi_main_roads',
            %d, %d, false));
        """ % (srid, int(origin_id), int(target_id))
    pg_cursor.execute(selectQuery) 
    result = pg_cursor.fetchall() 
    if len(result) > 0: 
        return json.dumps(result[0])
 
 
# Block 5: Get result of operation
path = get_shortest_path(source, target, srid)
 
pg_conn.close() 
 
# Block 6: Return result to the client 
print("Content-type: application/json") 
print() 
print(path)