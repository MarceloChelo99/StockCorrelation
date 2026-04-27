import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel,
  text,
  shape,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} from "@oai/artifact-tool";

const OUT = {
  pptx: "output/output.pptx",
  scratch: "scratch",
};

const W = 1920;
const H = 1080;

const C = {
  ink: "#17201B",
  muted: "#667166",
  faint: "#8C927F",
  paper: "#FFF8E8",
  cream: "#F4ECD9",
  sand: "#E4D2A8",
  moss: "#2F6858",
  mint: "#BEE8A6",
  blue: "#346F9D",
  clay: "#C7653D",
  gold: "#E1A93E",
  plum: "#5B4B6F",
  red: "#B94B48",
  line: "#D2C4A5",
  dark: "#101715",
};

const T = {
  eyebrow: { fontFamily: "Avenir Next", fontSize: 22, bold: true, color: C.moss },
  title: { fontFamily: "Avenir Next Condensed", fontSize: 62, bold: true, color: C.ink },
  subtitle: { fontFamily: "Avenir Next", fontSize: 25, color: C.muted },
  body: { fontFamily: "Avenir Next", fontSize: 25, color: C.ink },
  small: { fontFamily: "Avenir Next", fontSize: 17, color: C.muted },
  foot: { fontFamily: "Avenir Next", fontSize: 13, color: C.faint },
  metric: { fontFamily: "Avenir Next Condensed", fontSize: 66, bold: true, color: C.ink },
  metricSmall: { fontFamily: "Avenir Next Condensed", fontSize: 44, bold: true, color: C.ink },
  mono: { fontFamily: "Menlo", fontSize: 20, color: C.ink },
};

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
});

const slides = [];

function addSlide(render) {
  const slide = presentation.slides.add();
  slides.push(slide);
  render(slide, slides.length);
  return slide;
}

function base(slide, slideNo, children, opts = {}) {
  const bg = opts.bg || C.cream;
  slide.compose(
    layers({ name: `slide-${slideNo}-base`, width: fill, height: fill }, [
      shape({ name: `slide-${slideNo}-background`, width: fill, height: fill, fill: bg }),
      column(
        {
          name: `slide-${slideNo}-root`,
          width: fill,
          height: fill,
          padding: { x: 82, y: 58 },
          gap: 28,
        },
        children,
      ),
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

function titleStack(slideNo, eyebrow, titleValue, subtitleValue, width = 1320) {
  return column({ name: `slide-${slideNo}-title-stack`, width: fill, height: hug, gap: 10 }, [
    text(eyebrow, {
      name: `slide-${slideNo}-eyebrow`,
      width: fixed(620),
      height: hug,
      style: T.eyebrow,
    }),
    text(titleValue, {
      name: `slide-${slideNo}-title`,
      width: fixed(width),
      height: hug,
      style: T.title,
    }),
    text(subtitleValue, {
      name: `slide-${slideNo}-subtitle`,
      width: fixed(width),
      height: hug,
      style: T.subtitle,
    }),
  ]);
}

function footer(source) {
  return row({ name: "footer", width: fill, height: hug, align: "center", gap: 18 }, [
    rule({ name: "footer-rule", width: grow(1), stroke: C.line, weight: 1 }),
    text(source, { name: "source-rail", width: wrap(760), height: hug, style: T.foot }),
  ]);
}

function metric(value, label, fillColor = C.paper, accent = C.moss) {
  const isDarkMetric = fillColor === "#16221D" || fillColor === C.dark;
  const labelColor = isDarkMetric ? "#D8E1C5" : C.ink;
  return panel(
    {
      name: `metric-${label.slice(0, 12).replace(/[^a-z0-9]+/gi, "-")}`,
      width: fill,
      height: fixed(150),
      fill: fillColor,
      line: { style: "solid", width: 1, fill: C.line },
      borderRadius: "rounded-lg",
      padding: { x: 22, y: 14 },
    },
    column({ width: fill, height: fill, gap: 4 }, [
      text(value, { width: fill, height: hug, style: { ...T.metricSmall, color: accent } }),
      text(label, { width: fill, height: hug, style: { ...T.small, color: labelColor } }),
    ]),
  );
}

function callout(titleValue, bodyValue, accent = C.moss) {
  return row({ width: fill, height: hug, gap: 18, align: "start" }, [
    shape({ name: `callout-mark-${titleValue.slice(0, 8)}`, width: fixed(10), height: fixed(104), fill: accent }),
    column({ width: fill, height: hug, gap: 8 }, [
      text(titleValue, { width: fill, height: hug, style: { ...T.body, bold: true, color: C.ink } }),
      text(bodyValue, { width: fill, height: hug, style: { ...T.small, fontSize: 20 } }),
    ]),
  ]);
}

function miniTable(headers, rows, widths, opts = {}) {
  const headerFill = opts.headerFill || C.dark;
  const headerStyle = { ...T.small, fontSize: 17, bold: true, color: C.paper };
  const bodyStyle = { ...T.small, fontSize: opts.fontSize || 17, color: C.ink };
  const tracks = widths.map((w) => fixed(w));
  const rowGap = opts.rowGap ?? 6;
  const cells = [];
  cells.push(
    ...headers.map((h, i) =>
      panel(
        {
          name: `table-head-${i}`,
          width: fill,
          height: fixed(42),
          fill: headerFill,
          line: { style: "solid", width: 0, fill: headerFill },
          borderRadius: "rounded-sm",
          padding: { x: 10, y: 8 },
        },
        text(h, { width: fill, height: hug, style: headerStyle }),
      ),
    ),
  );
  rows.forEach((rowValues, r) => {
    rowValues.forEach((value, c) => {
      cells.push(
        panel(
          {
            name: `table-cell-${r}-${c}`,
            width: fill,
            height: fixed(opts.rowHeight || 52),
            fill: r % 2 === 0 ? "#FFF9EC" : "#F7EEDB",
            line: { style: "solid", width: 1, fill: "#E1D3B8" },
            borderRadius: "rounded-sm",
            padding: { x: 10, y: 8 },
          },
          text(String(value), { width: fill, height: hug, style: bodyStyle }),
        ),
      );
    });
  });
  return grid(
    {
      name: "mini-table",
      width: fill,
      height: hug,
      columns: tracks,
      columnGap: 6,
      rowGap,
      autoRows: "auto",
    },
    cells,
  );
}

function methodPill(label, body, accent) {
  return row({ width: fill, height: hug, gap: 14, align: "center" }, [
    panel(
      {
        width: fixed(86),
        height: fixed(58),
        fill: accent,
        line: { style: "solid", width: 0, fill: accent },
        borderRadius: "rounded-full",
        padding: { x: 10, y: 13 },
        align: "center",
        justify: "center",
      },
      text(label, {
        width: fill,
        height: hug,
        style: { ...T.small, fontSize: 18, bold: true, color: C.paper },
      }),
    ),
    text(body, { width: fill, height: hug, style: { ...T.body, fontSize: 24 } }),
  ]);
}

function nativeChart(name, type, config, height = 390) {
  return panel(
    {
      name: `${name}-shell`,
      width: fill,
      height: fixed(height + 34),
      fill: C.paper,
      line: { style: "solid", width: 1, fill: C.line },
      borderRadius: "rounded-lg",
      padding: { x: 18, y: 16 },
    },
    chart({
      name,
      chartType: type,
      width: fill,
      height: fill,
      config: {
        title: "",
        titlePlacement: "none",
        chartFill: { type: "solid", color: "FFF8E8" },
        plotAreaFill: { type: "solid", color: "FFF8E8" },
        hasLegend: true,
        legend: { position: "bottom", textStyle: { fontSize: 11 } },
        dataLabels: { showValue: false },
        ...config,
      },
    }),
  );
}

addSlide((slide, n) => {
  base(
    slide,
    n,
    [
      row({ width: fill, height: fill, gap: 48, align: "center" }, [
        column({ width: grow(1.25), height: fill, gap: 30, justify: "center" }, [
          text("Stock Embeddings", {
            name: "cover-title",
            width: wrap(930),
            height: hug,
            style: { ...T.title, fontSize: 94, color: C.paper },
          }),
          text("A decomposed similarity system for public companies", {
            name: "cover-subtitle",
            width: wrap(890),
            height: hug,
            style: { ...T.subtitle, fontSize: 31, color: "#D8E1C5" },
          }),
          rule({ width: fixed(320), stroke: C.mint, weight: 6 }),
          text(
            "What we have so far: SEC filing embeddings, price behavior, fundamentals, graph relationships, evaluations, and an exploratory dashboard.",
            { width: wrap(920), height: hug, style: { ...T.body, color: "#F2F1E7", fontSize: 27 } },
          ),
        ]),
        column({ width: fixed(520), height: fill, gap: 18, justify: "center" }, [
          metric("502", "tickers in the panel", "#16221D", C.mint),
          metric("87k", "monthly firm observations", "#16221D", C.mint),
          metric("4", "similarity views", "#16221D", C.mint),
          metric("61", "unit tests passing", "#16221D", C.mint),
        ]),
      ]),
      footer("Project artifacts through 2026-04-27; branch: explore."),
    ],
    { bg: C.dark },
  );
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "THESIS",
      "Company similarity is not one object",
      "The project now separates business language, trading behavior, lifecycle fundamentals, and relationship networks so each can be tested on the task it should actually explain.",
    ),
    row({ width: fill, height: fill, gap: 24 }, [
      column({ width: grow(1), height: fill, gap: 22 }, [
        methodPill("01", "Business view: historical MiniLM filing-section embeddings capture what firms say they do.", C.moss),
        methodPill("02", "Behavioral view: price-derived features capture how stocks move and trade.", C.blue),
        methodPill("03", "Growth view: XBRL fundamentals and valuation ratios capture lifecycle and maturity.", C.gold),
        methodPill("04", "Network view: disclosed relationships capture who firms compete with, buy from, sell to, or partner with.", C.clay),
      ]),
      panel(
        {
          width: fixed(560),
          height: fill,
          fill: C.paper,
          line: { style: "solid", width: 1, fill: C.line },
          borderRadius: "rounded-lg",
          padding: 30,
        },
        column({ width: fill, height: fill, gap: 22, justify: "center" }, [
          text("Current read", { width: fill, height: hug, style: { ...T.eyebrow, color: C.clay } }),
          text("Text finds economic meaning; GICS still wins short-horizon co-movement.", {
            width: fill,
            height: hug,
            style: { ...T.title, fontSize: 48 },
          }),
          text(
            "That split is the project’s useful insight: semantic similarity and risk similarity overlap, but they are not substitutes.",
            { width: fill, height: hug, style: { ...T.body, fontSize: 24 } },
          ),
        ]),
      ),
    ]),
    footer("Sources: report/current_implementation_and_results.md; report/decomposed_similarity_results.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "PIPELINE",
      "The codebase now follows the ETL shape the project spec wanted",
      "Raw SEC and price data flow into stable artifacts, then feature producers, assembly, models, evaluators, reports, and dashboard views.",
    ),
    grid(
      {
        width: fill,
        height: fill,
        columns: [fr(1), fr(1), fr(1), fr(1), fr(1)],
        columnGap: 14,
        rowGap: 18,
      },
      [
        metric("1", "ingest: SEC DB + prices", C.paper, C.moss),
        metric("2", "features: parquet groups", C.paper, C.blue),
        metric("3", "assembly: monthly PIT panel", C.paper, C.gold),
        metric("4", "model: PyTorch embeddings", C.paper, C.clay),
        metric("5", "evaluation + dashboard", C.paper, C.plum),
        callout("Feature producers", "price, event counts, historical text, fundamentals, valuation, network position", C.moss),
        callout("Model registry", "autoencoder, temporal autoencoder, PCA; legacy NumPy retained only for comparison", C.blue),
        callout("Evaluation registry", "clustering, peers, covariance, relationships, multi-view covariance, view comparison", C.gold),
        callout("Script layout", "pipeline stages, historical_text, decomposed, analysis, audit, temporal_validation", C.clay),
        callout("Reproducibility", "experiment dirs preserve config, embeddings, metrics, figures, logs", C.plum),
      ],
    ),
    footer("Sources: PROJECT_SPEC.md; report/project_structure_for_code_review.md; report/current_implementation_and_results.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "DATA LAYER",
      "The useful historical asset is now compact filing embeddings, not raw text hoarding",
      "The 10-K + 10-Q stream stores section-level embeddings and topic counts across time, so the dashboard can track language movement without carrying all source filings forward.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      column({ width: grow(1), height: fill, gap: 18 }, [
        miniTable(
          ["Artifact", "Rows", "Tickers", "Size / note"],
          [
            ["section embeddings", "115,549", "500", "277 MB"],
            ["topic counts", "125,359", "500", "2.65 MB"],
            ["text_historical features", "87,282", "502", "64 PCA dims"],
            ["fundamentals", "729,163", "500", "14 concepts"],
            ["relationship graph", "676", "265 src", "sparse, conservative"],
          ],
          [330, 150, 150, 300],
          { rowHeight: 58, fontSize: 18 },
        ),
        text("Completed text artifact currently covers 10-K and 10-Q filings since 2010; 8-K and other forms are supported next, but not fully embedded yet.", {
          width: fill,
          height: hug,
          style: { ...T.small, fontSize: 19 },
        }),
      ]),
      nativeChart(
        "section-coverage-chart",
        "bar",
        {
          categories: ["q_mda", "q_market", "q_legal", "q_risk", "mda", "risk", "business", "cyber"],
          series: [{ name: "section rows", values: [21399, 19532, 15773, 15180, 7387, 7302, 7232, 1470] }],
          hasLegend: false,
          barOptions: { direction: "column", grouping: "clustered", gapWidth: 70 },
          yAxis: { numberFormat: "0" },
          dataLabels: { showValue: false },
        },
        500,
      ),
    ]),
    footer("Sources: report/historical_text_trends.md; report/current_implementation_and_results.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "EMBEDDINGS",
      "Historical text materially improved the business view",
      "The strongest positive result remains semantic structure: company descriptions cluster into economically sensible groups and now beat the earlier text baseline on GICS NMI.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      nativeChart(
        "embedding-quality-chart",
        "bar",
        {
          categories: ["Legacy text", "Risk / price", "Historical text", "Temporal text"],
          series: [
            { name: "NMI vs GICS", values: [0.381, 0.171, 0.433, 0.412] },
            { name: "ARI vs GICS", values: [0.219, 0.052, 0.255, 0.269] },
          ],
          hasLegend: true,
          legend: { position: "bottom", textStyle: { fontSize: 12 } },
          barOptions: { direction: "column", grouping: "clustered", gapWidth: 90 },
          yAxis: { minimumScale: 0, maximumScale: 0.55, numberFormat: "0.00" },
        },
        510,
      ),
      column({ width: fixed(575), height: fill, gap: 18 }, [
        callout("Best semantic model", "Historical-text business AE: NMI 0.433, ARI 0.255.", C.moss),
        callout("Risk view behavior", "Price/risk embedding is weaker semantically, but stronger for return co-movement.", C.blue),
        callout("Qualitative examples", "META with WDAY/CRM/INTU; ETN with utilities; CBRE with alt-asset managers; SBUX with CPG brands.", C.clay),
      ]),
    ]),
    footer("Sources: report/current_implementation_and_results.md; experiments/*/metrics.json."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "LANGUAGE DRIFT",
      "The dashboard is already finding the AI regime shift",
      "The 2023-2026 filing stream shows AI language rising sharply, especially in risk factors, and the company-level movers are economically plausible.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      nativeChart(
        "ai-trend-chart",
        "line",
        {
          categories: ["2023", "2024", "2025", "2026*"],
          series: [
            { name: "Business AI score", values: [3.389, 5.693, 6.909, 7.112] },
            { name: "Risk-factor AI score", values: [1.576, 6.803, 9.988, 11.22] },
          ],
          hasLegend: true,
          legend: { position: "bottom", textStyle: { fontSize: 12 } },
          yAxis: { minimumScale: 0, maximumScale: 12, numberFormat: "0.0" },
          lineOptions: { smooth: false },
        },
        510,
      ),
      column({ width: fixed(575), height: fill, gap: 18 }, [
        text("Largest AI-language increases", { width: fill, height: hug, style: { ...T.eyebrow, color: C.clay } }),
        text("EPAM, ADBE, ADP, AMZN, EFX, PANW, NVDA, CDNS, NOW, INTU, ORCL, QCOM, MSFT, META.", {
          width: fill,
          height: hug,
          style: { ...T.title, fontSize: 43 },
        }),
        text("This is exactly what the dashboard should be good at: finding strategic language shifts outside the obvious sector buckets.", {
          width: fill,
          height: hug,
          style: { ...T.body, fontSize: 24 },
        }),
      ]),
    ]),
    footer("Sources: report/historical_text_trends.md. *2026 is partial local sample."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "DECOMPOSITION",
      "The four views are related, but not redundant",
      "Cross-view NMI is low enough to support the framework: each view organizes the universe differently.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      column({ width: grow(1), height: fill, gap: 18 }, [
        miniTable(
          ["View pair", "NMI"],
          [
            ["Behavioral / Business", "0.312"],
            ["Business / Network", "0.302"],
            ["Behavioral / Network", "0.257"],
            ["Business / Growth", "0.214"],
            ["Behavioral / Growth", "0.208"],
            ["Growth / Network", "0.197"],
          ],
          [470, 140],
          { rowHeight: 54, fontSize: 19 },
        ),
        text("Mean off-diagonal NMI: 0.248. Max: 0.312.", {
          width: fill,
          height: hug,
          style: { ...T.body, fontSize: 25, bold: true, color: C.moss },
        }),
      ]),
      column({ width: fixed(680), height: fill, gap: 16 }, [
        callout("Behavioral", "Most dynamic; useful as a trading-regime map from 2011 onward.", C.blue),
        callout("Business", "Strongest sector/thematic separation latest, but historical movie needs cleaner PIT text coverage.", C.moss),
        callout("Growth", "Lifecycle-oriented: margins, leverage, payout, valuation, capital intensity.", C.gold),
        callout("Network", "Stable structural map; current graph is static and sparse.", C.clay),
      ]),
    ]),
    footer("Sources: report/decomposed_similarity_results.md; report/view_cluster_dynamics_summary.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "PEERS",
      "GICS still wins return co-movement, but the horizon pattern is informative",
      "Business similarity becomes less negative at longer horizons; behavioral similarity is best around medium horizons.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      nativeChart(
        "peer-horizon-chart",
        "line",
        {
          categories: ["21d", "63d", "126d", "252d", "504d"],
          series: [
            { name: "Business gap vs GICS", values: [-0.272, -0.201, -0.197, -0.189, -0.177] },
            { name: "Risk gap vs GICS", values: [-0.1, -0.072, -0.071, -0.072, -0.079] },
          ],
          hasLegend: true,
          legend: { position: "bottom", textStyle: { fontSize: 12 } },
          yAxis: { minimumScale: -0.3, maximumScale: 0, numberFormat: "0.00" },
        },
        505,
      ),
      column({ width: fixed(575), height: fill, gap: 18 }, [
        callout("Do not oversell this", "Embedding peers do not beat GICS sub-industry peers on return correlation.", C.red),
        callout("But it tells a story", "Text peers capture structural similarity; price peers capture behavioral co-movement.", C.moss),
        callout("Next test", "Longer horizons, filing-date windows, and event-aware 8-K features should be the sharper tests.", C.blue),
      ]),
    ]),
    footer("Sources: report/current_implementation_and_results.md; report/decomposed_similarity_results.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "COVARIANCE",
      "Ledoit-Wolf remains the full-universe benchmark",
      "The direct multi-view factor covariance model is a clean negative result. That is useful: it tells us not to replace the statistical baseline wholesale.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      nativeChart(
        "covariance-chart",
        "bar",
        {
          categories: ["Sample", "Ledoit-Wolf", "Single-view", "Multi-view"],
          series: [{ name: "Annual variance", values: [0.00966, 0.00963, 0.01009, 0.01522] }],
          hasLegend: false,
          barOptions: { direction: "column", grouping: "clustered", gapWidth: 85 },
          yAxis: { minimumScale: 0, maximumScale: 0.017, numberFormat: "0.000" },
          dataLabels: { showValue: true, numberFormat: "0.000" },
        },
        505,
      ),
      column({ width: fixed(575), height: fill, gap: 18 }, [
        callout("Full universe", "Multi-view annual variance: 0.01522 vs Ledoit-Wolf 0.00963.", C.red),
        callout("Slice result", "Single-view embedding prior won Health Care, mid-liquidity, and Consumer Discretionary slices.", C.gold),
        callout("Implication", "Use Ledoit-Wolf as base; route or blend embedding signals only where they repeatedly help.", C.moss),
      ]),
    ]),
    footer("Sources: report/multiview_covariance_results.md; report/current_implementation_and_results.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "TEMPORAL AE",
      "Temporal smoothing cleans trajectories, not events",
      "The temporal autoencoder passed stability and preserved semantic structure, but failed the AI-drift and event-alignment bars.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      column({ width: grow(1), height: fill, gap: 18 }, [
        miniTable(
          ["Diagnostic", "Result", "Read"],
          [
            ["Stability", "PASS", "stable firms move less"],
            ["NMI preservation", "PASS", "cross-sectional structure holds"],
            ["AI drift", "FAIL", "1.5x vs 2x target"],
            ["Event alignment", "FAIL", "0% pass rate"],
          ],
          [220, 130, 430],
          { rowHeight: 62, fontSize: 19 },
        ),
        text("Best sweep candidate: lambda=0.5, alpha=2.0, NMI 0.496, but still no event-alignment success.", {
          width: fill,
          height: hug,
          style: { ...T.small, fontSize: 19 },
        }),
      ]),
      column({ width: fixed(650), height: fill, gap: 18 }, [
        metric("83,549", "consecutive training pairs", C.paper, C.moss),
        metric("0.408", "within-firm variance ratio", C.paper, C.blue),
        metric("15 / 15", "stability sweep passes", C.paper, C.gold),
      ]),
    ]),
    footer("Sources: report/temporal_autoencoder_results.md; report/temporal_validation/hyperparameter_sweep.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "DASHBOARD",
      "The dashboard has become the project’s discovery surface",
      "The most useful productized piece is not a backtest result. It is the ability to watch filings, themes, maps, and sector signals evolve interactively.",
    ),
    grid(
      {
        width: fill,
        height: fill,
        columns: [fr(1), fr(1), fr(1)],
        columnGap: 18,
        rowGap: 18,
      },
      [
        callout("Filing Browser", "browse parsed raw filing sections and source metadata", C.moss),
        callout("Historical Text", "topic trends, semantic drift, company and sector language deltas", C.blue),
        callout("Similarity Shifts", "theme loadings, dynamic labels, company movement over time", C.gold),
        callout("Market Map", "stocks as 2D points with smooth step-through time controls", C.clay),
        callout("Sector Outlook", "walk-forward sector excess-return model vs equal-weight S&P", C.plum),
        panel(
          {
            width: fill,
            height: fixed(190),
            fill: C.dark,
            line: { style: "solid", width: 0, fill: C.dark },
            borderRadius: "rounded-lg",
            padding: 22,
          },
          column({ width: fill, height: fill, gap: 10, justify: "center" }, [
            text("Run it", { width: fill, height: hug, style: { ...T.eyebrow, color: C.mint } }),
            text(".venv/bin/streamlit run apps/raw_filing_browser/app.py --server.port 8502", {
              width: fill,
              height: hug,
              style: { ...T.mono, color: C.paper, fontSize: 17 },
            }),
          ]),
        ),
      ],
    ),
    footer("Sources: apps/raw_filing_browser/app.py; report/current_implementation_and_results.md."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "SECTOR OUTLOOK",
      "A transparent sector-relative model is now in the dashboard",
      "This is a small walk-forward ridge model: price momentum and volatility plus sector-mean valuation/growth signals predict sector excess return versus the equal-weight S&P universe.",
    ),
    row({ width: fill, height: fill, gap: 28 }, [
      column({ width: fixed(600), height: fill, gap: 18 }, [
        metric("0.132", "mean rank IC", C.paper, C.moss),
        metric("1.41%", "top-minus-bottom spread", C.paper, C.blue),
        metric("60.9%", "top-sector hit rate", C.paper, C.gold),
      ]),
      column({ width: grow(1), height: fill, gap: 20 }, [
        miniTable(
          ["Latest top-ranked sectors", "Model read"],
          [
            ["Communication Services", "highest current score"],
            ["Financials", "near top of latest ranking"],
            ["Health Care", "positive relative outlook"],
            ["Consumer Staples", "defensive + valuation mix"],
            ["Information Technology", "still top-five"],
          ],
          [360, 430],
          { rowHeight: 58, fontSize: 19 },
        ),
        text("Important caveat: valuation coverage is improving, but cash-flow-derived features need a fundamentals refresh before treating fundamentals as fully represented.", {
          width: fill,
          height: hug,
          style: { ...T.small, fontSize: 19 },
        }),
      ]),
    ]),
    footer("Sources: src/applications/sector_relative_outlook.py; dashboard smoke test metrics from local artifacts."),
  ]);
});

addSlide((slide, n) => {
  base(slide, n, [
    titleStack(
      n,
      "NEXT",
      "The right next step is data refresh, not model glitter",
      "The codebase is now ready for sharper inputs. The highest-value work is refreshing fundamentals, adding event-aware filing text, and making temporal movement validate around filing dates.",
    ),
    grid(
      {
        width: fill,
        height: fill,
        columns: [fr(1), fr(1)],
        columnGap: 28,
        rowGap: 22,
      },
      [
        callout("Refresh fundamentals", "rerun XBRL companyfacts with cash flow and expanded share-count concepts so valuation features fill in", C.gold),
        callout("Add 8-K text to temporal input", "Item 1.01, 2.01, 2.02, 7.01, 8.01 should capture events the 10-K/10-Q view misses", C.clay),
        callout("Validate around filings", "measure velocity around filing dates, not only calendar events whose disclosure channel may differ", C.blue),
        callout("Keep covariance conservative", "Ledoit-Wolf remains the base; test embedding-informed routing only in slices where evidence repeats", C.moss),
      ],
    ),
    footer("Sources: report/current_implementation_and_results.md; report/temporal_autoencoder_results.md."),
  ]);
});

await mkdir(OUT.scratch, { recursive: true });
await mkdir("output", { recursive: true });

for (let i = 0; i < slides.length; i += 1) {
  const slideNo = String(i + 1).padStart(2, "0");
  const pngBlob = await slides[i].export({ format: "png" });
  await writeFile(path.join(OUT.scratch, `source_slide_${slideNo}.png`), Buffer.from(await pngBlob.arrayBuffer()));
  const layoutBlob = await slides[i].export({ format: "layout" });
  await writeFile(path.join(OUT.scratch, `source_slide_${slideNo}.layout.json`), await layoutBlob.text());
}

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save(OUT.pptx);

const pptxBytes = await readFile(OUT.pptx);
const imported = await PresentationFile.importPptx(pptxBytes);
for (let i = 0; i < imported.slides.count; i += 1) {
  const slideNo = String(i + 1).padStart(2, "0");
  const importedSlide = imported.slides.getItem(i);
  const pngBlob = await importedSlide.export({ format: "png" });
  await writeFile(path.join(OUT.scratch, `pptx_slide_${slideNo}.png`), Buffer.from(await pngBlob.arrayBuffer()));
  const layoutBlob = await importedSlide.export({ format: "layout" });
  await writeFile(path.join(OUT.scratch, `pptx_slide_${slideNo}.layout.json`), await layoutBlob.text());
}

console.log(JSON.stringify({ pptx: OUT.pptx, slides: slides.length }, null, 2));
