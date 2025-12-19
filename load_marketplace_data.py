# load_marketplace_data.py
"""
Script to load marketplace intelligence data into the RAG store.
Run this to add more marketplace data for better analysis.
"""

from rag_store import add_marketplace_data

# Extended marketplace data
marketplace_intelligence = [
    {
        "text": "Food delivery apps market: $150B globally. DoorDash, Uber Eats, Grubhub lead. Average order value: $25. Delivery fees: $2-5. Success factors: fast delivery, restaurant partnerships, user ratings.",
        "source": "Market Research",
        "category": "Food Delivery",
        "metadata": {"market_size": "$150B", "avg_order": "$25"}
    },
    {
        "text": "E-commerce marketplace apps growing 25% YoY. Amazon, eBay, Shopify stores. Mobile commerce: 60% of total e-commerce. Key features: one-click ordering, AR try-on, personalized recommendations.",
        "source": "E-commerce Report",
        "category": "E-commerce",
        "metadata": {"growth": "25% YoY", "mobile_share": "60%"}
    },
    {
        "text": "Ride-sharing market: $200B+. Uber, Lyft, Didi dominate. Average fare: $15-20. Surge pricing controversial but increases revenue 2-3x. Electric vehicles transition underway.",
        "source": "Transportation Analytics",
        "category": "Transportation",
        "metadata": {"market_size": "$200B+", "avg_fare": "$15-20"}
    },
    {
        "text": "Social media apps: Facebook, Instagram, TikTok, Snapchat. Daily active users: 4B+. Advertising revenue: $150B. Key metrics: engagement rate, time spent, algorithmic feed optimization.",
        "source": "Social Media Report",
        "category": "Social Media",
        "metadata": {"dau": "4B+", "ad_revenue": "$150B"}
    },
    {
        "text": "Streaming services market: $50B+. Netflix, Disney+, HBO Max. Average subscription: $6.99/month. Original content crucial for retention. Binge-watching changes viewing habits.",
        "source": "Entertainment Analytics",
        "category": "Streaming",
        "metadata": {"market_size": "$50B+", "avg_price": "$6.99/month"}
    },
    {
        "text": "Fintech apps revolutionizing banking. Venmo, Cash App, Robinhood lead. P2P payments: $1.2T annually. Key challenges: regulation, security, user trust. Mobile banking adoption: 80%.",
        "source": "Fintech Report",
        "category": "Financial Technology",
        "metadata": {"p2p_volume": "$1.2T", "mobile_adoption": "80%"}
    }
]

if __name__ == "__main__":
    success = add_marketplace_data(marketplace_intelligence)
    if success:
        print("✅ Marketplace data successfully added to RAG store!")
        print(f"Added {len(marketplace_intelligence)} marketplace intelligence entries")
    else:
        print("❌ Failed to add marketplace data")