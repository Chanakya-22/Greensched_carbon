from jobs.job import JobGenerator


def test_arrivals_are_whole_hours_inside_window():
    window_hours = 168
    jobs = JobGenerator.generate(
        num_jobs=1000,
        window_hours=window_hours,
        seed=42,
    )

    assert all(isinstance(job.arrival_time, int) for job in jobs)
    assert all(0 <= job.arrival_time < window_hours for job in jobs)


def test_deadline_slack_is_within_requested_bounds():
    min_slack = 0.5
    max_slack = 3.0
    jobs = JobGenerator.generate(
        num_jobs=1000,
        window_hours=168,
        min_slack_hours=min_slack,
        max_slack_hours=max_slack,
        seed=42,
    )

    for job in jobs:
        slack = (
            job.deadline
            - job.arrival_time
            - job.duration_estimate_hours
        )
        assert min_slack <= slack <= max_slack


def test_jobs_are_sorted_by_arrival():
    jobs = JobGenerator.generate(num_jobs=1000, window_hours=168, seed=42)

    arrivals = [job.arrival_time for job in jobs]
    assert arrivals == sorted(arrivals)


def test_same_seed_generates_identical_jobs():
    first = JobGenerator.generate(num_jobs=100, window_hours=168, seed=42)
    second = JobGenerator.generate(num_jobs=100, window_hours=168, seed=42)

    assert first == second


def test_cpu_ratio_is_within_five_percent_over_large_sample():
    requested_ratio = 0.7
    jobs = JobGenerator.generate(
        num_jobs=1000,
        window_hours=168,
        cpu_ratio=requested_ratio,
        seed=42,
    )
    actual_ratio = sum(job.job_type == "cpu_openmp" for job in jobs) / len(jobs)

    assert abs(actual_ratio - requested_ratio) <= 0.05
