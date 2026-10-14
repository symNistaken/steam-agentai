import sys
from pathlib import Path
import unittest
import json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app
from services.agentic_service import AgenticRecommendationService


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()

    def test_home_health_and_stats(self):
        for path in ('/','/api/health','/api/db/stats'):
            self.assertEqual(self.client.get(path).status_code,200)

    def test_bad_body_types_are_client_errors(self):
        for body in ([1], {'steam_input':4}, {'steam_input':None}):
            self.assertEqual(self.client.post('/api/analyze',json=body).status_code,400)

    def test_missing_ids_do_not_contact_services(self):
        with patch('services.steam_service.SteamService.resolve_steam_id') as resolve:
            for path in ('/api/analyze','/api/agentic/recommend','/api/matchmaking','/api/rag/context','/api/xai/explain'):
                self.assertEqual(self.client.post(path,json={}).status_code,400)
            resolve.assert_not_called()

    def test_invalid_scores_are_rejected(self):
        for score in ('nan','inf','invalid',2,-1):
            self.assertEqual(self.client.post('/api/xai/explain',json={'steam_id':'76561197960265730','game_name':'Example','recommendation_score':score}).status_code,400)

    def test_rag_context_does_not_reuse_another_profiles_library(self):
        for first_id in (100, 200):
            games=[{'appid':first_id+i, 'name':f'Space adventure {i}', 'description':'Space exploration adventure', 'playtime_forever':120} for i in range(8)]
            with patch('services.steam_service.SteamService.get_player_summary', return_value={'personaname':'Test'}), patch('services.steam_service.SteamService.get_owned_games',return_value=games):
                response=self.client.post('/api/rag/context',json={'steam_id':'76561197960265730','max_context':3})
            self.assertEqual(response.status_code,200)
            context=response.get_json()['context_games']
            self.assertTrue(context)
            self.assertTrue(all(first_id <= game['app_id'] < first_id+8 for game in context))


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.service=AgenticRecommendationService()

    def test_malformed_llm_json_is_marked_as_fallback(self):
        for text in ('not JSON', '{"recommendations": [42]}'):
            self.assertEqual(self.service._parse_llm_response(text)['source'],'fallback')

    def test_valid_response_excludes_owned_games(self):
        response=json.dumps({'recommendations':[
            {'name':'Owned','appid':1,'reason':'Example'},
            {'name':'New','appid':2,'reason':'Example'},
        ]})
        parsed=self.service._parse_llm_response(response,[{'appid':1}])
        self.assertEqual(parsed['source'],'groq')
        self.assertEqual([game['appid'] for game in parsed['recommendations']],[2])
