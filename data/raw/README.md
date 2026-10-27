# `data/raw/`

Place the dataset here:

```
data/raw/hospital_dataset_preprocessed.csv
```

The path is configured in `config/config.yaml` (`data.raw_csv`). If you rename the
file, update that key.

## Expected columns

The pipeline expects the columns present in the provided dataset, including (not
exhaustive):

- **Identity / timing:** `appointment_id`, `patient_id`, `booking_timestamp`,
  `appointment_date`, `appointment_time`, `time_slot`, `day_of_week`, `hour_of_day`
- **Booking-time features:** `lead_time_hours`, `lead_time_days`, `is_weekend`,
  `is_peak_hour`, `is_holiday`, `priority_level`, `previous_no_show_count`,
  `previous_late_count`, `total_visits_count`, `active_doctors_count`,
  `slot_arrival_count`, `department`
- **Check-in / service:** `check_in_time`, `arrival_delay`,
  `queue_length_at_arrival`, `service_start_time`, `service_end_time`,
  `service_time`, `waiting_time`, `queue_id`, `doctor_id`
- **Targets:** `no_show`, `arrival_delay`, `waiting_time`

Dates are parsed **day-first** (DD-MM-YYYY).

> This system is **appointment-only** — it does not model walk-ins. `priority_level`
> distinguishes emergency (`1`) from routine (`0`) bookings.

To use real clinic data later, drop the real CSV here (matching these columns) and
re-run the three pipeline scripts — no code changes required.
