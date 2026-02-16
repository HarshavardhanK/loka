"""Tests for Loka agent tools."""



class TestTrajectoryTool:
    """Tests for the trajectory computation tool."""

    def test_hohmann_transfer_earth_mars(self):
        """Test Hohmann transfer calculation for Earth-Mars."""
        from loka.tools.trajectory_tool import TrajectoryTool

        tool = TrajectoryTool()

        # Approximate circular orbit radii (km)
        # Earth: ~1 AU, Mars: ~1.524 AU
        earth_r = 1.496e8  # km
        mars_r = 1.524 * 1.496e8  # km

        result = tool.execute(
            transfer_type="hohmann",
            origin_position=[earth_r, 0, 0],
            target_position=[mars_r, 0, 0],
            central_body="sun",
        )

        assert result.success
        output = result.output

        # Check delta-V is reasonable (should be ~5-6 km/s total for Earth-Mars)
        assert 3 < output["total_delta_v_km_s"] < 10

        # Check transfer time (should be ~250-260 days for Hohmann)
        assert 200 < output["transfer_time_days"] < 300

    def test_invalid_transfer_type(self):
        """Test that invalid transfer type returns error."""
        from loka.tools.trajectory_tool import TrajectoryTool

        tool = TrajectoryTool()

        result = tool.execute(
            transfer_type="invalid",
            origin_position=[1e8, 0, 0],
            target_position=[2e8, 0, 0],
        )

        assert not result.success
        assert "Unknown transfer type" in result.error

    def test_lambert_requires_transfer_time(self):
        """Test that Lambert transfer requires transfer_time parameter."""
        from loka.tools.trajectory_tool import TrajectoryTool

        tool = TrajectoryTool()

        result = tool.execute(
            transfer_type="lambert",
            origin_position=[1e8, 0, 0],
            target_position=[2e8, 0, 0],
        )

        assert not result.success
        assert "transfer_time" in result.error


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        from loka.tools.base import Tool, ToolRegistry, ToolResult

        class DummyTool(Tool):
            name = "dummy"
            description = "A dummy tool"

            @property
            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            def execute(self, **kwargs):
                return ToolResult(success=True, output="dummy output")

        registry = ToolRegistry()
        registry.register(DummyTool())

        tool = registry.get("dummy")
        assert tool is not None
        assert tool.name == "dummy"

    def test_execute_unknown_tool(self):
        """Test executing an unknown tool returns error."""
        from loka.tools.base import ToolRegistry

        registry = ToolRegistry()

        result = registry.execute("unknown_tool")

        assert not result.success
        assert "Unknown tool" in result.error
