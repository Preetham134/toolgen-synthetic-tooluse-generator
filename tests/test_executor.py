from __future__ import annotations

import unittest

from toolgen.executor.mock_executor import MockToolExecutor
from toolgen.executor.state import ConversationState
from toolgen.registry.models import Endpoint


def make_endpoint(endpoint_id: str, api_name: str, description: str = "", tool_name: str = "demo") -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        tool_name=tool_name,
        api_name=api_name,
        category="general",
        description=description,
    )


class TestConversationState(unittest.TestCase):
    def test_state_next_id_increments(self) -> None:
        state = ConversationState()
        self.assertEqual(state.next_id("hotel"), "hotel_001")
        self.assertEqual(state.next_id("hotel"), "hotel_002")


class TestMockExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = MockToolExecutor()
        self.state = ConversationState()

    def test_search_execution_returns_results_and_updates_state(self) -> None:
        endpoint = make_endpoint("hotels.search", "search_hotels", "Search hotels by city", tool_name="hotels")
        output = self.executor.execute(endpoint, {"city": "Paris"}, self.state)

        self.assertIn("results", output)
        self.assertEqual(output["count"], 2)
        self.assertTrue(all(item["id"].startswith("hotel_") for item in output["results"]))
        self.assertEqual(len(self.state.get_entities("hotel")), 2)
        self.assertEqual(self.state.get_slot("city"), "Paris")

    def test_booking_reuses_prior_search_result_id(self) -> None:
        search_endpoint = make_endpoint("hotels.search", "search_hotels", "Find hotels", tool_name="hotels")
        book_endpoint = make_endpoint("hotels.book", "book_hotel", "Book hotel", tool_name="hotels")

        search_output = self.executor.execute(search_endpoint, {"city": "Paris"}, self.state)
        hotel_id = search_output["results"][0]["id"]
        booking_output = self.executor.execute(book_endpoint, {"hotel_id": hotel_id}, self.state)

        self.assertEqual(booking_output["hotel_id"], hotel_id)
        self.assertEqual(booking_output["status"], "confirmed")

    def test_detail_lookup_returns_existing_entity_when_id_known(self) -> None:
        search_endpoint = make_endpoint("hotels.search", "search_hotels", "Find hotels", tool_name="hotels")
        detail_endpoint = make_endpoint("hotels.details", "get_hotel_details", "Get hotel details", tool_name="hotels")

        search_output = self.executor.execute(search_endpoint, {"city": "Paris"}, self.state)
        hotel_id = search_output["results"][0]["id"]
        detail_output = self.executor.execute(detail_endpoint, {"hotel_id": hotel_id}, self.state)

        self.assertEqual(detail_output["id"], hotel_id)

    def test_fallback_execution_returns_echo(self) -> None:
        endpoint = make_endpoint("misc.run", "do_magic", "Unknown operation")
        output = self.executor.execute(endpoint, {"value": 3}, self.state)

        self.assertEqual(output["status"], "ok")
        self.assertEqual(output["echo"], {"value": 3})


if __name__ == "__main__":
    unittest.main()
