# How to Run the Application

1. **Install Dependencies**  
   Use the following command to install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
   It's recommended to use a virtual environment to avoid conflicts with other Python projects.  

2. **Run the Application**  
   Start the app using one of the following commands:  
   - Using Uvicorn:  
     ```bash
     uvicorn main:app --reload
     ```  
   - Alternatively:  
     ```bash
     fastapi dev main.py
     ```  

3. **Access the Documentation**  
   Open the Swagger UI for API documentation by navigating to:  
   [http://localhost:PORT_NUMBER/docs](http://localhost:PORT_NUMBER/docs)  
   Replace `PORT_NUMBER` with the port specified when starting the application (default is `8000`).
