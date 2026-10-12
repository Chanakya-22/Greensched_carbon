"""
Scheduling policies for job placement across regions.
Each policy implements a different optimization strategy.
"""
from typing import List, Dict, Union


DEFAULT_REGION = "region-balanced"


def fifo_ondemand_policy(
    job,
    current_hour: int,
    region_options: List[Dict]
) -> Union[Dict, str]:
    """
    Model an organisation that always deploys to its home region regardless
    of price or carbon.
    
    The default home region is selected when available; otherwise the first
    available region is used.
    
    Args:
        job: Job object with job_id, job_type, arrival_time, deadline, etc.
        current_hour: Current simulation hour
        region_options: List of dicts with region, carbon_intensity, price_per_hr
        
    Returns:
        Home region dict if available, otherwise the first option, or "wait"
        if no options are available
    """
    if region_options:
        return next(
            (
                region
                for region in region_options
                if region["region"] == DEFAULT_REGION
            ),
            region_options[0],
        )
    return "wait"


def cheapest_first_policy(
    job,
    current_hour: int,
    region_options: List[Dict]
) -> Union[Dict, str]:
    """
    Cheapest First Policy (Cost Optimization)
    
    Picks the region with the lowest price_per_hr among options that still
    allow the job to meet its deadline if started this hour. If no region
    can meet the deadline, picks the cheapest available region anyway.
    
    Optimization target: Minimize total cost
    
    Args:
        job: Job object with deadline and duration_estimate_hours
        current_hour: Current simulation hour
        region_options: List of dicts with region, carbon_intensity, price_per_hr
        
    Returns:
        Cheapest region dict (deadline-feasible preferred), or "wait"
    """
    if not region_options:
        return "wait"
    
    # Calculate completion hour if job starts now
    completion_hour = current_hour + job.duration_estimate_hours
    
    # Filter regions that allow meeting deadline
    deadline_feasible = [
        r for r in region_options
        if completion_hour <= job.deadline
    ]
    
    # If any region meets deadline, pick cheapest among them
    if deadline_feasible:
        return min(deadline_feasible, key=lambda r: r["price_per_hr"])
    
    # Otherwise pick cheapest available region
    return min(region_options, key=lambda r: r["price_per_hr"])


def cleanest_first_policy(
    job,
    current_hour: int,
    region_options: List[Dict]
) -> Union[Dict, str]:
    """
    Cleanest First Policy (Carbon Optimization)
    
    Picks the region with the lowest carbon_intensity among options that
    still allow the job to meet its deadline if started this hour. If no
    region can meet the deadline, picks the cleanest available region anyway.
    
    Optimization target: Minimize total carbon emissions
    
    Args:
        job: Job object with deadline and duration_estimate_hours
        current_hour: Current simulation hour
        region_options: List of dicts with region, carbon_intensity, price_per_hr
        
    Returns:
        Cleanest region dict (deadline-feasible preferred), or "wait"
    """
    if not region_options:
        return "wait"
    
    # Calculate completion hour if job starts now
    completion_hour = current_hour + job.duration_estimate_hours
    
    # Filter regions that allow meeting deadline
    deadline_feasible = [
        r for r in region_options
        if completion_hour <= job.deadline
    ]
    
    # If any region meets deadline, pick cleanest among them
    if deadline_feasible:
        return min(deadline_feasible, key=lambda r: r["carbon_intensity"])
    
    # Otherwise pick cleanest available region
    return min(region_options, key=lambda r: r["carbon_intensity"])


def joint_policy(
    job,
    current_hour: int,
    region_options: List[Dict]
) -> Union[Dict, str]:
    """
    Joint Policy (Balanced Cost + Carbon Optimization)
    
    Normalizes price and carbon_intensity across available regions to a 0-1 scale,
    computes a weighted score (0.5 * normalized_price + 0.5 * normalized_carbon),
    and picks the lowest-scoring option among those that meet the deadline.
    If no region can meet the deadline, picks the option with lowest weighted score.
    
    This policy balances cost and environmental impact equally.
    
    Optimization target: Minimize weighted combination of cost and carbon
    
    Args:
        job: Job object with deadline and duration_estimate_hours
        current_hour: Current simulation hour
        region_options: List of dicts with region, carbon_intensity, price_per_hr
        
    Returns:
        Best region dict (lowest joint score), or "wait"
    """
    if not region_options:
        return "wait"
    
    # Calculate completion hour if job starts now
    completion_hour = current_hour + job.duration_estimate_hours
    
    # Filter regions that allow meeting deadline
    deadline_feasible = [
        r for r in region_options
        if completion_hour <= job.deadline
    ]
    
    # If deadline-feasible options exist, score only those
    # Otherwise score all available options
    candidates = deadline_feasible if deadline_feasible else region_options
    
    # Normalize price and carbon to 0-1 scale
    prices = [r["price_per_hr"] for r in candidates]
    carbons = [r["carbon_intensity"] for r in candidates]
    
    min_price = min(prices)
    max_price = max(prices)
    min_carbon = min(carbons)
    max_carbon = max(carbons)
    
    # Avoid division by zero
    price_range = max_price - min_price if max_price > min_price else 1.0
    carbon_range = max_carbon - min_carbon if max_carbon > min_carbon else 1.0
    
    # Compute normalized scores and pick best
    best_region = None
    best_score = float("inf")
    
    for region in candidates:
        normalized_price = (region["price_per_hr"] - min_price) / price_range
        normalized_carbon = (region["carbon_intensity"] - min_carbon) / carbon_range
        
        # Weighted score: 50% price, 50% carbon
        score = 0.5 * normalized_price + 0.5 * normalized_carbon
        
        if score < best_score:
            best_score = score
            best_region = region
    
    return best_region if best_region else "wait"


def make_wait_policy(
    objective: str,
    w_cost: float = 0.5,
    margin: float = 0.02,
):
    """Create a forecast-aware policy that waits for worthwhile improvements."""
    if objective not in {"cost", "carbon", "joint"}:
        raise ValueError("objective must be 'cost', 'carbon', or 'joint'.")
    if not 0 <= w_cost <= 1:
        raise ValueError("w_cost must be between 0 and 1.")
    if margin < 0:
        raise ValueError("margin must be non-negative.")

    def wait_policy(job, current_hour: int, region_options, *, forecast):
        del job, region_options
        if not forecast:
            return "wait"

        if objective == "joint":
            prices = [candidate["price_per_hr"] for candidate in forecast]
            carbons = [candidate["carbon_intensity"] for candidate in forecast]
            min_price, max_price = min(prices), max(prices)
            min_carbon, max_carbon = min(carbons), max(carbons)
            price_range = max_price - min_price
            carbon_range = max_carbon - min_carbon

            def score(candidate):
                norm_price = (
                    (candidate["price_per_hr"] - min_price) / price_range
                    if price_range
                    else 0.0
                )
                norm_carbon = (
                    (candidate["carbon_intensity"] - min_carbon) / carbon_range
                    if carbon_range
                    else 0.0
                )
                return w_cost * norm_price + (1 - w_cost) * norm_carbon

        elif objective == "cost":
            score = lambda candidate: candidate["price_per_hr"]
        else:
            score = lambda candidate: candidate["carbon_intensity"]

        best_now = min(
            (
                candidate
                for candidate in forecast
                if candidate["start_hour"] == current_hour
            ),
            key=score,
            default=None,
        )
        best_future = min(
            (
                candidate
                for candidate in forecast
                if candidate["start_hour"] > current_hour
            ),
            key=score,
            default=None,
        )

        if best_now is None:
            return min(forecast, key=score)

        if best_future is not None:
            if objective == "joint":
                should_wait = score(best_future) + margin < score(best_now)
            else:
                should_wait = (
                    score(best_future) * (1 + margin) < score(best_now)
                )
            if should_wait:
                return "wait"

        return {
            "region": best_now["region"],
            "carbon_intensity": best_now["carbon_intensity"],
            "price_per_hr": best_now["price_per_hr"],
        }

    wait_policy.__name__ = f"{objective}_wait_policy"
    wait_policy.needs_forecast = True
    return wait_policy


cheapest_wait_policy = make_wait_policy("cost")
cleanest_wait_policy = make_wait_policy("carbon")
joint_wait_policy = make_wait_policy("joint")
