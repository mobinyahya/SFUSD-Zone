 SFUSD Zoning Dashboard Plan                                                                       
                                                                                                   
 Overview                                                                                          
                                                                                                   
 A web dashboard for community stakeholders to interact with school zoning solutions. Features an  
 interactive map (left), chat interface (right), and analytics graphs (bottom). Users select from  
 clustered solution groups via dropdown before viewing.                                            
                                                                                                   
 Architecture                                                                                      
                                                                                                   
 website/                                                                                          
 ├── backend/                                                                                      
 │   ├── app.py              # FastAPI server                                                      
 │   ├── data_loader.py      # Load solutions, graphs, geodata                                     
 │   └── requirements.txt                                                                          
 ├── frontend/                                                                                     
 │   ├── index.html                                                                                
 │   ├── style.css                                                                                 
 │   └── app.js              # Leaflet map, chat UI, charts                                        
 └── data/                                                                                         
     └── sf_blockgroups.geojson  # Pre-converted geodata                                           
                                                                                                   
 Data Pipeline                                                                                     
                                                                                                   
 1. Data Sources                                                                                   
                                                                                                   
 - Solutions CSV:                                                                                  
 /home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/llm_bg_runs/recursive_metrics_flattened.csv  
 - Zone Dict: {path}/zone_dict_BlockGroup_0.json - maps graph node IDs to zone IDs                 
 - Graph: /share/data/school_choice/Optimization/Zones/Graphs/BlockGroup_0.pickle - contains       
 node.area_id for blockgroup mapping                                                               
 - Shapefile:                                                                                      
 /share/data/school_choice/shapefiles/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp          
                                                                                                   
 2. Data Flow                                                                                      
                                                                                                   
 1. Load graph pickle to build node_id -> blockgroup_id mapping                                    
 2. Load zone_dict to get node_id -> zone_id                                                       
 3. Combine: blockgroup_id -> zone_id                                                              
 4. Convert shapefile to GeoJSON with blockgroup IDs                                               
 5. Merge zone assignments into GeoJSON properties                                                 
                                                                                                   
 Implementation Steps                                                                              
                                                                                                   
 Step 1: Backend Setup (app.py)                                                                    
                                                                                                   
 FastAPI with CORS enabled. Endpoints:                                                             
 - GET /api/clusters - return clustered solutions using LLM.exploration.clusters module:           
   - Load CSV, compute Pareto frontier, cluster solutions                                          
   - Return cluster labels (e.g. "Better economic diversity; accepts higher commute") and          
 representative solution path per cluster                                                          
 - GET /api/solution/{path} - get zone assignment + demographics for specific solution path        
 - GET /api/geojson - serve pre-converted GeoJSON                                                  
 - POST /api/chat - placeholder returning static message                                           
                                                                                                   
 Step 2: Data Loader (data_loader.py)                                                              
                                                                                                   
 - load_solutions_csv() - parse CSV with metric columns + path                                     
 - load_graph() - load pickle, extract node_id -> area_id mapping from graph nodes                 
 - load_zone_dict(path) - load JSON, map node IDs to blockgroup IDs using graph                    
 - convert_shapefile_to_geojson() - one-time conversion script                                     
 - get_zone_demographics(zone_dict, graph) - aggregate demographics per zone from graph node       
 attributes                                                                                        
 - get_clusters() - use LLM.exploration.pareto and LLM.exploration.clusters to:                    
   a. load_solutions() and normalize_metrics()                                                     
   b. compute_pareto_frontier()                                                                    
   c. vectorize_solutions() and cluster_solutions()                                                
   d. compute_cluster_directions() for labels                                                      
   e. get_representative_solution() for each cluster                                               
                                                                                                   
 Step 3: Frontend - Initial Load                                                                   
                                                                                                   
 - On page load, fetch /api/clusters                                                               
 - Show dropdown with cluster options (label + solution count)                                     
 - No map visible until user selects a cluster                                                     
 - When cluster selected, fetch /api/solution/{path} for that cluster's representative             
 - Display map with zones                                                                          
                                                                                                   
 Step 4: Frontend Map (Leaflet)                                                                    
                                                                                                   
 - Interactive map of SF with blockgroup boundaries                                                
 - Color-coded by zone assignment (use colors from Constants.zone_colors)                          
 - Hover: highlight blockgroup, show tooltip with:                                                 
   - BlockGroup ID                                                                                 
   - Zone ID                                                                                       
   - Zone demographics (FRL%, ethnicity breakdown, student count)                                  
 - Click: expand details panel or center on zone                                                   
                                                                                                   
 Step 5: Chat Interface                                                                            
                                                                                                   
 - Right panel with message history                                                                
 - Input bar at bottom                                                                             
 - Sends to /api/chat, receives static message: "Agent connection coming soon. Currently viewing:  
 [cluster label]"                                                                                  
 - Response displays in message list                                                               
                                                                                                   
 Step 6: Analytics Charts (Chart.js)                                                               
                                                                                                   
 Bottom panel with 2 charts:                                                                       
 - Bar chart: Demographics breakdown by zone (FRL%, ethnic percentages)                            
 - Bar chart: Student count per zone                                                               
                                                                                                   
 Key Files to Create                                                                               
                                                                                                   
 1. website/backend/app.py - FastAPI server                                                        
 2. website/backend/data_loader.py - Data processing utilities                                     
 3. website/backend/requirements.txt - Dependencies                                                
 4. website/frontend/index.html - Page structure with dropdown, map, chat, charts                  
 5. website/frontend/style.css - Grid layout styling                                               
 6. website/frontend/app.js - Map, chat, charts logic                                              
 7. website/data/sf_blockgroups.geojson - Pre-converted geodata (generated on first run)           
                                                                                                   
 Dependencies                                                                                      
                                                                                                   
 Backend (requirements.txt):                                                                       
 fastapi                                                                                           
 uvicorn                                                                                           
 pandas                                                                                            
 geopandas                                                                                         
 networkx                                                                                          
 scikit-learn                                                                                      
                                                                                                   
 Frontend (CDN):                                                                                   
 - Leaflet.js 1.9                                                                                  
 - Chart.js 4.x                                                                                    
                                                                                                   
 Future Agent Connection (Overview)                                                                
                                                                                                   
 To connect the live agent from LLM/exploration/zoning_agent.py:                                   
 1. Add WebSocket endpoint /ws/chat                                                                
 2. Instantiate ZoningAgent(csv_path) per session                                                  
 3. On message: response = agent.chat(user_message)                                                
 4. Parse response for solution path, send map update to client                                    
 5. Agent tools (tighten_filter, select_cluster) already return solution paths                     
                                                                                                   
 Verification                                                                                      
                                                                                                   
 1. Generate GeoJSON: python data_loader.py (run once)                                             
 2. Start backend: cd website/backend && uvicorn app:app --reload --port 8000                      
 3. Open http://localhost:8000                                                                     
 4. Verify cluster dropdown populated                                                              
 5. Select a cluster - map loads with colored zones                                                
 6. Hover over blockgroups - tooltip shows demographics                                            
 7. Type in chat - static response appears                                                         
 8. Verify charts show zone breakdown     