from oil_engine_core import load_config, run_engine_once
import time
import traceback


def main():
    print("Starting Oil Daily Engine...")

    try:
        config = load_config("settings.daily.yaml")
        print("Config loaded successfully")
    except Exception as e:
        print("Failed to load config:", e)
        return

    poll_interval = config.get("runtime", {}).get("poll_interval_seconds", 60)

    while True:
        try:
            print("\nRunning engine cycle...")
            result = run_engine_once(config)

            skipped = result.get("skipped")
            reason = result.get("skip_reason")
            print(f"Cycle Result | skipped={skipped} | reason={reason}")

            ranked_df = result.get("ranked_df")
            if ranked_df is not None and not ranked_df.empty:
                preview_cols = [
                    c for c in [
                        "contract_ticker",
                        "event_ticker",
                        "resolution_time_et",
                        "decision_state",
                        "action",
                        "selected_side",
                        "selected_edge",
                    ]
                    if c in ranked_df.columns
                ]
                if preview_cols:
                    print(ranked_df[preview_cols].head(5).to_string(index=False))
            else:
                print("No ranked candidates this cycle")

        except Exception:
            print("Engine cycle failed:")
            traceback.print_exc()

        print(f"Sleeping for {poll_interval} seconds...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
