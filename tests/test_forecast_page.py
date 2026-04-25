"""Regression tests for forecast page behavior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ui.pages.forecast_page import ForecastPage


def _fake_result() -> SimpleNamespace:
    history = np.linspace(1.0, 4.0, 16, dtype=np.float64)
    prediction = np.linspace(5.0, 6.0, 24, dtype=np.float64)
    return SimpleNamespace(history=history, prediction=prediction)


def _hourly_series(size: int = 64) -> pd.Series:
    return pd.Series(
        np.linspace(1.0, 3.0, size, dtype=np.float64),
        index=pd.date_range("2026-04-01 00:00:00", periods=size, freq="h"),
        name="WHE",
    )


def test_repeated_same_selection_reuses_existing_history_record(qapp, monkeypatch) -> None:
    page = ForecastPage()
    page.set_current_user("alice")
    page.load_prepared_series(_hourly_series())

    saved_calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        page._service,
        "run_forecast",
        lambda model_id, horizon, values: _fake_result(),
    )
    monkeypatch.setattr(
        "ui.pages.forecast_page.save_prediction_record",
        lambda **kwargs: saved_calls.append(
            (kwargs["model_id"], kwargs["horizon_key"], kwargs["username"])
        )
        or 7,
    )

    page._on_run_clicked()
    page._on_run_clicked()

    assert saved_calls == [("dc_itransformer_ampds", "1d", "alice")]


def test_result_summary_includes_peak_hour_tip_for_one_day_forecast(
    qapp, monkeypatch
) -> None:
    page = ForecastPage()
    page.load_prepared_series(_hourly_series())

    prediction = np.linspace(10.0, 20.0, 24, dtype=np.float64)
    prediction[7] = 99.0

    monkeypatch.setattr(
        page._service,
        "run_forecast",
        lambda model_id, horizon, values: SimpleNamespace(
            history=np.linspace(1.0, 4.0, 16, dtype=np.float64),
            prediction=prediction,
        ),
    )

    page._on_run_clicked()

    summary = page._result_label.text()
    assert "预测均值：" in summary
    assert "用电高峰提醒：预计 1 天内高峰出现在第 8 个小时" in summary


def test_result_summary_includes_peak_day_tip_for_seven_day_forecast(
    qapp, monkeypatch
) -> None:
    page = ForecastPage()
    page.load_prepared_series(_hourly_series())
    page._horizon_combo.setCurrentIndex(1)

    prediction = np.linspace(20.0, 30.0, 168, dtype=np.float64)
    prediction[95] = 120.0

    monkeypatch.setattr(
        page._service,
        "run_forecast",
        lambda model_id, horizon, values: SimpleNamespace(
            history=np.linspace(1.0, 4.0, 16, dtype=np.float64),
            prediction=prediction,
        ),
    )

    page._on_run_clicked()

    summary = page._result_label.text()
    assert "预测均值：" in summary
    assert "用电高峰提醒：预计高峰出现在 4 天后" in summary


def test_forecast_page_offers_thirty_day_horizon(qapp) -> None:
    page = ForecastPage()

    horizons = [
        (page._horizon_combo.itemText(index), page._horizon_combo.itemData(index))
        for index in range(page._horizon_combo.count())
    ]

    assert ("30天", "30d") in horizons


def test_result_summary_includes_peak_day_tip_for_thirty_day_forecast(
    qapp, monkeypatch
) -> None:
    page = ForecastPage()
    page.load_prepared_series(_hourly_series())
    page._horizon_combo.setCurrentIndex(2)

    prediction = np.linspace(20.0, 30.0, 720, dtype=np.float64)
    prediction[287] = 180.0

    monkeypatch.setattr(
        page._service,
        "run_forecast",
        lambda model_id, horizon, values: SimpleNamespace(
            history=np.linspace(1.0, 4.0, 16, dtype=np.float64),
            prediction=prediction,
        ),
    )

    page._on_run_clicked()

    summary = page._result_label.text()
    assert "预测均值：" in summary
    assert "用电高峰提醒：预计高峰出现在 12 天后" in summary


def test_hover_lookup_returns_prediction_point_metadata(qapp, monkeypatch) -> None:
    page = ForecastPage()
    page.load_prepared_series(_hourly_series())

    prediction = np.linspace(20.0, 30.0, 24, dtype=np.float64)
    monkeypatch.setattr(
        page._service,
        "run_forecast",
        lambda model_id, horizon, values: SimpleNamespace(
            history=np.linspace(1.0, 4.0, 64, dtype=np.float64),
            prediction=prediction,
        ),
    )

    page._on_run_clicked()
    page._chart_canvas.draw()

    first_prediction_x = page._hover_targets[1]["x"][0]
    first_prediction_y = page._hover_targets[1]["y"][0]
    display_x, display_y = page._ax.transData.transform(
        np.array([[first_prediction_x, first_prediction_y]])
    )[0]

    hit = page._find_nearest_hover_point(float(display_x), float(display_y))

    assert hit is not None
    assert hit["series_name"] == "预测值"
    assert hit["index_label"] == "2026-04-03 16:00"
    assert hit["value"] == prediction[0]
