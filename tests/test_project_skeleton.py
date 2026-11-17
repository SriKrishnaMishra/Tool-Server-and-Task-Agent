from bodhan_agent import __name__ as agent_package
from bodhan_mcp import __name__ as mcp_package


def test_packages_are_importable() -> None:
    assert agent_package == "bodhan_agent"
    assert mcp_package == "bodhan_mcp"
