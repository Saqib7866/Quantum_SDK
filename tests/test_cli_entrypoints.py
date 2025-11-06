from qx.cli import run_main, report_main


def test_run_main_smoke():
    # default run_main uses bell example and sim-local backend
    rc = run_main([])
    assert rc == 0


def test_report_main_smoke():
    rc = report_main(["--list"])
    assert rc == 0
