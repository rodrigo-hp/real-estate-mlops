wrk.method = "POST"
wrk.body = '{"type":"casa","sector":"vitacura","net_usable_area":152.0,"net_area":250.0,"n_rooms":3.0,"n_bathroom":3.0,"latitude":-33.3794,"longitude":-70.5447}'
wrk.headers["Content-Type"] = "application/json"
wrk.headers["x-api-key"] = os.getenv("API_KEY")