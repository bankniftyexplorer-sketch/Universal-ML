import os
import sys
import time
import json
import pickle
import glob


def print_header(title):
    print(
        f"\n================================================================================"
    )
    print(f"  {title}")
    print(
        f"================================================================================"
    )


def run_dashboard():
    print_header("TOON UNIVERSAL-ML : INSTITUTIONAL TERMINAL")

    registry_path = "portfolio_sleeve_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
            sleeves = data.get("sleeves", {})
            print(f"\n[ 1. PORTFOLIO ADMISSION STATUS ]")
            print(
                f"{'SYMBOL':<12} {'LANE':<6} {'STATUS':<10} {'VARIANT':<8} {'SHARPE':<8} {'PROFIT.F':<8} {'TRADES'}"
            )
            print("-" * 75)
            for key, details in sorted(sleeves.items()):
                sym = details.get("symbol", "N/A")
                lane = details.get("lane", "N/A")
                enabled = details.get("enabled", False)
                status = "🟢 ACTIVE" if enabled else "🔴 BLOCKED"
                variant = details.get("selected_variant", "None")
                sharpe = details.get("sharpe")
                pf = details.get("profit_factor")
                trds = details.get("total_trades", 0)

                s_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
                pf_str = f"{pf:.2f}" if pf is not None else "N/A"

                print(
                    f"{sym:<12} {lane:<6} {status:<10} {str(variant):<8} {s_str:<8} {pf_str:<8} {trds}"
                )
    else:
        print("\n>> No Portfolio Registry found.")

    print(f"\n[ 2. LIVE ML TARGET RADAR (LATEST META-VERDICTS) ]")
    print(
        f"{'SYMBOL':<12} {'WINNING STRATEGY':<20} {'EDGE (SHARPE)':<15} {'DIRECTION':<10} {'SIGNAL':<12} {'CONF'}"
    )
    print("-" * 88)

    verdict_files = sorted(glob.glob("*/*_meta_verdict.pkl"))
    for vf in verdict_files:
        try:
            with open(vf, "rb") as f:
                v_data = pickle.load(f)
                sym = v_data.get("symbol", "N/A")
                winner = v_data.get("winner", "N/A")

                metrics = v_data.get("metrics", {}).get(winner, {})
                sharpe = metrics.get("sharpe")
                sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"

                sig = v_data.get("signal", {})
                direction = sig.get("direction", "N/A")
                strength = sig.get("signal_strength", "N/A")
                conf = sig.get("confidence")
                conf_str = f"{conf * 100:.1f}%" if conf is not None else "N/A"

                if strength not in ["NO_TRADE", "SIDELINED"]:
                    strength = f"🔥 {strength}"
                elif strength == "SIDELINED":
                    strength = f"⏸️ {strength}"
                else:
                    strength = f"   {strength}"

                print(
                    f"{sym:<12} {winner:<20} {sharpe_str:<15} {direction:<10} {strength:<12} {conf_str}"
                )
        except Exception:
            pass

    print(f"\n[ 3. REAL-TIME INFERENCE TARGETS (TP / SL) ]")
    print(
        f"{'SYMBOL':<12} {'LANE':<6} {'DIRECTION':<10} {'SIGNAL':<12} {'STOP LOSS':<15} {'TAKE PROFIT (1)':<18} {'TAKE PROFIT (2)':<18}"
    )
    print("-" * 105)

    signal_files = sorted(glob.glob("*/*_live_signal_*.json"))
    for sf in signal_files:
        try:
            with open(sf, "r") as jf:
                s_data = json.load(jf)
                sym = s_data.get("symbol", "N/A")
                lane = s_data.get("lane", "N/A")
                direction = s_data.get("direction", "N/A")
                strength = s_data.get("signal", "N/A")
                sl = s_data.get("stop_loss")
                tp1 = s_data.get("tp1")
                tp2 = s_data.get("tp2")

                if strength not in ["NO_TRADE", "SIDELINED"]:
                    strength = f"⚡ {strength}"
                else:
                    strength = f"   {strength}"

                sl_str = f"{sl:,.2f}" if sl else "N/A"
                tp1_str = f"{tp1:,.2f}" if tp1 else "N/A"
                tp2_str = f"{tp2:,.2f}" if tp2 else "N/A"

                print(
                    f"{sym:<12} {lane:<6} {direction:<10} {strength:<12} {sl_str:<15} {tp1_str:<18} {tp2_str:<18}"
                )
        except Exception:
            pass

    print()


if __name__ == "__main__":
    if "--live" in sys.argv:
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                run_dashboard()
                print("\n[LIVE MODE] Press Ctrl+C to exit. Refreshing in 5 seconds...")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nExiting Live Dashboard.")
    else:
        run_dashboard()
