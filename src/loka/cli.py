"""
Loka CLI - Command line interface for the Loka agent.
"""

import argparse
import sys


def main():
    """Main entry point for the Loka CLI."""
    parser = argparse.ArgumentParser(
        description="Loka - Agentic AI for Astrophysics Navigation",
        prog="loka",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--model",
        type=str,
        default="loka-v1",
        help="Model to use for chat",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query celestial body position")
    query_parser.add_argument("body", type=str, help="Celestial body name")
    query_parser.add_argument("--epoch", type=str, help="Time of query (ISO format)")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Plan a mission")
    plan_parser.add_argument("--origin", type=str, required=True, help="Origin body")
    plan_parser.add_argument("--destination", type=str, required=True, help="Destination body")
    plan_parser.add_argument("--departure", type=str, help="Departure window start")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "chat":
        print("Chat mode not yet implemented.")
        print(f"Would load model: {args.model}")

    elif args.command == "query":
        from datetime import datetime

        from loka.astro.ephemeris import EphemerisManager

        eph = EphemerisManager()
        epoch = args.epoch or datetime.utcnow().isoformat()

        try:
            pos, vel = eph.get_state(args.body, epoch)
            print(f"Body: {args.body}")
            print(f"Epoch: {epoch}")
            print(f"Position (km): x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            print(f"Velocity (km/s): vx={vel[0]:.6f}, vy={vel[1]:.6f}, vz={vel[2]:.6f}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "plan":
        print("Mission planning not yet implemented.")
        print(f"Would plan: {args.origin} -> {args.destination}")

    elif args.command == "serve":
        print(f"Starting server on {args.host}:{args.port}")
        print("Server not yet implemented.")


if __name__ == "__main__":
    main()
