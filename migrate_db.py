import sqlite3

def migrate_db():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        # Add role column to users table
        c.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'doctor'")
        conn.commit()
        print("Database migrated successfully: added role column to users table")
    except Exception as e:
        print(f"Error migrating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_db()