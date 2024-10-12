# Image classification "in-the-wild" 

- Neurosünök:
  - Wiederschitz Diána, Neptun kód: E0YKJT
  - Oroszki Marietta, Neptun kód: CZYYIF

## Projekt leírás
A Kaggle-n találtunk sakkfiguráról gyűjtött adatokat, innen jött az ötlet hogy egy olyan osztályozót szeretnénk tanítani, ami a bástyát, futót és huszárt meg tudja különböztetni. Ehhez a talált adatot python kód segítségével letöltöttük, hogy ezen tudjuk tanítani a neurális hálót, később pedig az általunk készített képeken szeretnénk tesztelni.


## Files and Functions in the Repository
- `requirements.txt`: A projekthez szükséges Python csomagok és függőségek.
- `Dockerfile`: A konténerizált környezet definiálása a futtatáshoz.
- `data_acquisition.py` / `data_acquisition.ipynb`: Az adatforrás letöltéséhez és előkészítéséhez használt szkript vagy Jupyter notebook.
- `data_preparation.py` / `data_preparation.ipynb`: Az adatok előkészítése és felosztása tréning, validációs és teszt adatokra.
- `model_training.py` / `model_training.ipynb`: A modell betanítása és kiértékelése.
- `Dockerfile`: A Docker-kép definiálására használt fájl, amely lehetővé teszi a környezet futtatását konténerben.

## Related Works
- [Kapcsolódó kutatási anyagok, GitHub repository-k, blogbejegyzések stb.]
  - [Papír 1]: [link]
  - [GitHub repo]: [link]

## How to Build and Run
1. **Környezet beállítása**:
   - Telepítsd a szükséges függőségeket: 
     ```bash
     pip install -r requirements.txt
     ```

2. **Konténer építése és futtatása**:
   - Építsd meg a Docker konténert:
     ```bash
     docker build -t project-container .
     ```
   - Futtasd a konténert:
     ```bash
     docker run -it project-container
     ```

3. **Adatok letöltése és előkészítése**:
   - Futtasd a `data_acquisition.py` szkriptet az adatok letöltéséhez:
     ```bash
     python data_acquisition.py
     ```

4. **Modell betanítása és kiértékelése**:
   - Futtasd a `model_training.py` szkriptet a modell betanításához:
     ```bash
     python model_training.py
     ```

## Notes
- A projekt kódfuttatása Docker konténerben történik a **Dockerfile** segítségével.
- A letöltött adatokat a `data` mappába mentjük.
