from __future__ import annotations

import unittest

from toolgen.graph.builder import build_graph
from toolgen.graph.sampler import sample_chain
from toolgen.registry.models import Endpoint, Parameter


def make_endpoint(
    endpoint_id: str,
    tool_name: str,
    api_name: str,
    category: str,
    description: str,
    input_params: list[Parameter] | None = None,
    required_params: list[str] | None = None,
    output_schema: dict | None = None,
) -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        tool_name=tool_name,
        api_name=api_name,
        category=category,
        description=description,
        input_params=input_params or [],
        required_params=required_params or [],
        output_schema=output_schema or {},
    )


class TestGraphBuilder(unittest.TestCase):
    def test_graph_builder_creates_nodes_and_edges(self) -> None:
        endpoints = [
            make_endpoint(
                endpoint_id="hotels.search",
                tool_name="hotels",
                api_name="search_hotels",
                category="travel",
                description="Search hotels by city",
                output_schema={"properties": {"hotel_id": {"type": "string"}, "city": {"type": "string"}}},
            ),
            make_endpoint(
                endpoint_id="hotels.book",
                tool_name="hotels",
                api_name="book_hotel",
                category="travel",
                description="Book hotel details",
                input_params=[Parameter(name="hotel_id", required=True)],
                required_params=["hotel_id"],
            ),
            make_endpoint(
                endpoint_id="weather.get",
                tool_name="weather",
                api_name="get_forecast",
                category="weather",
                description="Get forecast details",
            ),
        ]

        graph = build_graph(endpoints)

        self.assertEqual(graph.summary["num_nodes"], 3)
        self.assertGreaterEqual(graph.summary["num_edges"], 1)
        self.assertIn("hotels.search", graph.adjacency)
        hotel_edges = graph.adjacency["hotels.search"]
        self.assertTrue(any(edge["target"] == "hotels.book" for edge in hotel_edges))
        matching_edge = next(edge for edge in hotel_edges if edge["target"] == "hotels.book")
        self.assertIn("same_family", matching_edge["relation_type"])
        self.assertIn("same_tool", matching_edge["relation_type"])
        self.assertIn("entity_id_overlap", matching_edge["relation_type"])
        self.assertIn("search_to_action", matching_edge["relation_type"])

    def test_graph_builder_handles_empty_registry(self) -> None:
        graph = build_graph([])
        self.assertEqual(graph.summary["num_nodes"], 0)
        self.assertEqual(graph.summary["num_edges"], 0)
        self.assertEqual(graph.nodes, [])
        self.assertEqual(graph.edges, [])
        self.assertEqual(graph.adjacency, {})

    def test_graph_builder_prefers_same_family_edges_over_unrelated_edges(self) -> None:
        endpoints = [
            make_endpoint(
                endpoint_id="travel_hotels.search",
                tool_name="travel_hotels",
                api_name="hotel_search",
                category="travel",
                description="Search hotels",
                output_schema={"properties": {"hotel_id": {"type": "string"}}},
            ),
            make_endpoint(
                endpoint_id="travel_hotels.book",
                tool_name="travel_hotels",
                api_name="hotel_book",
                category="travel",
                description="Book hotel",
                input_params=[Parameter(name="hotel_id", required=True)],
                required_params=["hotel_id"],
            ),
            make_endpoint(
                endpoint_id="shopping_orders.create",
                tool_name="shopping_orders",
                api_name="order_create",
                category="shopping",
                description="Create order",
                input_params=[Parameter(name="product_id", required=True)],
                required_params=["product_id"],
            ),
        ]
        graph = build_graph(endpoints)
        hotel_edges = {edge["target"]: edge for edge in graph.adjacency["travel_hotels.search"]}
        self.assertIn("travel_hotels.book", hotel_edges)
        self.assertNotIn("shopping_orders.create", hotel_edges)


class TestGraphSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(
            [
                make_endpoint(
                    endpoint_id="travel_hotels.search",
                    tool_name="travel_hotels",
                    api_name="search_hotels",
                    category="travel",
                    description="Find hotels and list options",
                    output_schema={"hotel_id": {"type": "string"}, "properties": {"hotel_id": {"type": "string"}}},
                ),
                make_endpoint(
                    endpoint_id="travel_hotels.details",
                    tool_name="travel_hotels",
                    api_name="hotel_details",
                    category="travel",
                    description="Get hotel details",
                    input_params=[Parameter(name="hotel_id", required=True)],
                    required_params=["hotel_id"],
                ),
                make_endpoint(
                    endpoint_id="travel_hotels.book",
                    tool_name="travel_bookings",
                    api_name="book_hotel",
                    category="travel",
                    description="Book hotel details",
                    input_params=[Parameter(name="hotel_id", required=True)],
                    required_params=["hotel_id"],
                ),
                make_endpoint(
                    endpoint_id="weather.get",
                    tool_name="weather",
                    api_name="get_forecast",
                    category="weather",
                    description="Get forecast details",
                ),
            ]
        )

    def test_sampler_returns_exact_chain_length_when_possible(self) -> None:
        chain = sample_chain(self.graph, chain_length=3, seed=7)
        self.assertEqual(len(chain), 3)

    def test_sampler_respects_required_category_when_possible(self) -> None:
        chain = sample_chain(self.graph, chain_length=2, required_category="travel", seed=7)
        self.assertEqual(len(chain), 2)
        self.assertTrue(all(node["category"] == "travel" for node in chain))

    def test_sampler_respects_min_distinct_tools_when_possible(self) -> None:
        chain = sample_chain(self.graph, chain_length=3, min_distinct_tools=2, seed=7)
        self.assertEqual(len(chain), 3)
        self.assertGreaterEqual(len({node["tool_name"] for node in chain}), 2)

    def test_sampler_prefers_same_family_continuation_when_available(self) -> None:
        graph = build_graph(
            [
                make_endpoint(
                    endpoint_id="travel_hotels.search",
                    tool_name="travel_hotels",
                    api_name="hotel_search",
                    category="travel",
                    description="Search hotels",
                    output_schema={"properties": {"hotel_id": {"type": "string"}}},
                ),
                make_endpoint(
                    endpoint_id="travel_hotels.details",
                    tool_name="travel_hotels",
                    api_name="hotel_details",
                    category="travel",
                    description="Get hotel details",
                    input_params=[Parameter(name="hotel_id", required=True)],
                    required_params=["hotel_id"],
                ),
                make_endpoint(
                    endpoint_id="food_restaurants.book",
                    tool_name="food_restaurants",
                    api_name="restaurant_book",
                    category="food",
                    description="Reserve restaurant",
                    input_params=[Parameter(name="restaurant_id", required=True)],
                    required_params=["restaurant_id"],
                ),
            ]
        )
        chain = sample_chain(graph, chain_length=2, seed=42)
        self.assertEqual([node["endpoint_id"] for node in chain], ["travel_hotels.search", "travel_hotels.details"])


if __name__ == "__main__":
    unittest.main()
