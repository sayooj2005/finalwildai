import sqlite3
for db in ['intrusion.db', 'wildlife.db']:
    print(f"--- Checking {db} ---")
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for t in tables:
            t_name = t[0]
            print(f"Table: {t_name}")
            try:
                cursor.execute(f"SELECT * FROM {t_name} LIMIT 5")
                rows = cursor.fetchall()
                for r in rows:
                    print(f"  {r}")
            except:
                pass
        conn.close()
    except Exception as e:
        print(f"Error checking {db}: {e}")
