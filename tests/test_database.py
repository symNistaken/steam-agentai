"""Offline checks for a fresh checkout without the private database cache."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


# Load the SQLite module directly: package __init__ also imports AI dependencies.
module_path = Path(__file__).resolve().parents[1] / 'database' / 'db_manager.py'
spec = importlib.util.spec_from_file_location('sqlite_database', module_path)
database = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database)


class DatabaseTests(unittest.TestCase):
    def test_fresh_database_and_persistent_upsert(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.db'
            manager = database.DatabaseManager(path)
            self.assertEqual(manager.get_stats()['total_users'], 0)
            self.assertTrue(manager.upsert_user(
                'test-user', 'Oyuncu', games=[
                    {'appid': 1, 'name': 'Played', 'playtime_forever': 120},
                    {'appid': 2, 'name': 'Unplayed', 'playtime_forever': 0},
                ]))
            reopened = database.DatabaseManager(path)
            user = reopened.get_user('test-user')
            self.assertEqual(user['personaname'], 'Oyuncu')
            self.assertEqual(user['game_count'], 1)
            self.assertEqual(user['total_playtime_hours'], 2)
            self.assertTrue(reopened.upsert_user('test-user', 'Updated'))
            self.assertEqual(reopened.get_stats()['total_users'], 1)
            self.assertEqual(reopened.get_user('test-user')['personaname'], 'Updated')

    def test_game_cache_limits_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = database.DatabaseManager(Path(directory) / 'cache.db')
            games = [{'appid': index, 'playtime_forever': index} for index in range(61)]
            optimized = manager.optimize_game_data(games)
            self.assertEqual(len(optimized), 50)
            self.assertEqual(optimized[0]['playtime'], 60)
            self.assertEqual(optimized[-1]['playtime'], 11)


if __name__ == '__main__':
    unittest.main()
