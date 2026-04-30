# Dashboard Interpretation Note

The dashboard should not ask viewers to stare at a cloud of points and infer meaning by eye. The maps and heatmaps are useful only when paired with evidence tables and examples.

## What Is Actually Valuable

- The cluster/map view is useful as an exploratory index: it helps pick companies, themes, and movement paths worth investigating.
- The map is not, by itself, evidence that the model found economic structure. PCA can make random high-dimensional structure look visually interesting.
- The valuable evidence is the companion table: top stocks, sector mix, dominant loadings, filing-fragment labels, and supplier/customer examples.
- The strongest classroom use is to pick one group and explain it with concrete companies and disclosures, not to claim the whole scatter plot is meaningful by visual inspection.

## How To Explain The Market Map

The market map is a compressed view of theme memberships. Nearby points have similar learned-theme loading vectors in the selected view. Because it is a 2D PCA projection, distances are approximate and should be treated as a navigation aid.

The right sequence for a demo is:

1. Pick the business or network view.
2. Pick a focus company.
3. Read the focus panel: top theme memberships, nearest peers by the full loading vector, and supplier/customer links.
4. Only then use the scatter plot to show where that company and its nearest peers sit in the broader market.

The dashboard now intentionally treats the focused peer table as the primary evidence. The map highlights the selected company and its closest theme-loading peers, so the visual answers a specific question instead of showing an undirected cloud of colored dots.

Projection method matters:

- PCA is fast, deterministic, and more stable for showing movement through time.
- UMAP can make local neighborhoods easier to see, but it can also exaggerate separation and make global distances less interpretable.
- The dashboard therefore computes nearest peers in the original theme-loading space regardless of whether the map projection is PCA or UMAP.

## What The Heatmap Is For

The sector-year heatmap is now treated as an optional diagnostic. It can show broad topic diffusion, such as AI language spreading from Information Technology into Utilities or Industrials, but it is not a cluster-validation exhibit.

The more useful output is the "Sector Topic Takeaways" table, which directly lists:

- sectors with the highest recent topic language
- sectors with the largest increase over time

## Recommended Class Framing

Say this plainly:

"The visualization is not the result. The result is that the embeddings let us organize companies into interpretable similarity views, and the dashboard lets us inspect those views. We validate the views with metrics, filing evidence, and examples. The scatter plot is how we navigate the structure, not proof by itself."
