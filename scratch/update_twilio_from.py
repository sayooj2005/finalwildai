import sqlite3
db_path = 'wildlife.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

from_num = '+12605975946'

cursor.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)", ('twilio_from_number', from_num))

conn.commit()
print(f"Twilio From Number {from_num} updated in wildlife.db")
conn.close()
