import React, { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

/**
 * Solar Sense - Solar Flare Prediction Dashboard
 * This is the only frontend file, which fetches data from the backend
 * and displays it in a responsive chart and hourly strip.
 * The chart shows the predicted solar flare classes for each hour of the day.
 * The hourly strip shows the time and class for each hour.
 * The app starts in dark mode, but can be toggled to light mode.
 * The app also shows the current time and date.
 * 
 * The app frontend currently does sklearn pipeline not pytorch pipeline.
 * There is also a section for a daily solar movie (171 Å) with overlays for bright regions.
 * And a description of the predicted and observed peak classes.
 * The app is responsive and works well on both desktop and mobile devices.
 */


/* ---------- helpers ---------- */

// ---- API base resolution (safe for webpack, CRA, and Vite) ----
const RENDER_API = "https://solarsense-api.onrender.com";

const API_BASE =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) ||
  (typeof window !== "undefined" && window.__API_BASE__) ||
  ((location.hostname === "localhost" || location.hostname === "127.0.0.1")
    ? "http://localhost:8000"
    : RENDER_API);
console.log("API_BASE =", API_BASE);

// ---- JSON fetch helper with timeout ----
const fetchJSON = async (url, ms = 90000) => {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const r = await fetch(url, { signal: ctrl.signal, mode: "cors" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally {
    clearTimeout(timer);
  }
};

const todayUTC = () => new Date().toISOString().slice(0, 10);
const shiftDay = (iso, d) => {
  const t = new Date(iso + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + d);
  return t.toISOString().slice(0, 10);
};
const longDate = (iso) =>
  new Date(iso + "T00:00:00Z").toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });

const makeDummy = () => {
  const cls = ["A", "B", "C", "M", "X"];
  return Array.from({ length: 24 }, (_, i) => ({
    time: `${i % 12 || 12}${i < 12 ? "AM" : "PM"}`,
    level: `${cls[Math.floor(Math.random() * 5)]}-Class`,
    classIndex: Math.floor(Math.random() * 5),
  }));
};

const sci = new Intl.NumberFormat("en", {
  notation: "scientific",
  maximumFractionDigits: 0,
});

// interpret local midnight, then convert to UTC YYYY-MM-DD
const localDayToUTCISO = (isoLocal) => {
  const d = new Date(isoLocal + "T00:00:00");
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 10);
};

// Retry helper to survive Render cold starts / slow first byte
async function warmFetch(url, opts = {}, tries = 6) {
  let delay = 1200;
  for (let i = 0; i < tries; i++) {
    const firstByteCtrl = new AbortController();
    const combinedSignal = opts.signal
      ? (AbortSignal.any ? AbortSignal.any([firstByteCtrl.signal, opts.signal])
                         : firstByteCtrl.signal)
      : firstByteCtrl.signal;

    try {
      const t = setTimeout(() => firstByteCtrl.abort(), 60000);
      const res = await fetch(url, { ...opts, signal: combinedSignal, mode: "cors", cache: "no-store" });
      clearTimeout(t);
      if (res.ok) return res;
    } catch (e) {
      // if the caller aborted, bubble it
      if (e?.name === "AbortError" && opts.signal?.aborted) throw e;
    }

    await new Promise(r => setTimeout(r, delay));
    delay = Math.min(delay * 1.8, 8000);
  }
  throw new Error("API not responding");
}


const getSummary = async (isoUtcDay, opts={}) => {
  const r = await warmFetch(`${API_BASE}/summary/${isoUtcDay}`, opts);
  return r.json();
};

// Build NASA GSFC daily AIA-171 movie URL for the given UTC date (YYYY-MM-DD).
const gsfcAIA171UrlFor = (isoUtcDay) => {
  const y = isoUtcDay.slice(0, 4);
  const m = isoUtcDay.slice(5, 7);
  const d = isoUtcDay.slice(8, 10);
  return `https://sdo.gsfc.nasa.gov/assets/img/dailymov/${y}/${m}/${d}/${y}${m}${d}_1024_0171.mp4`;
};

// SWPC "latest" as a final safety fallback
const SWPC_AIA171_LATEST_MP4 = "https://services.swpc.noaa.gov/images/animations/sdo-aia-171/latest.mp4";



// Use the wrapper for all calls
// const getForecast = async (isoUtcDay) => {
//   const r = await warmFetch(`${API_BASE}/forecast/${isoUtcDay}`);
//   return r.json();
// };

// const getSdo = async (isoUtcDay) => {
//   const r = await warmFetch(`${API_BASE}/sdo/${isoUtcDay}`);
//   return r.json();
// };

const getSdoLite = async (isoUtcDay, opts={}) => {
  const r = await warmFetch(`${API_BASE}/sdo-lite/${isoUtcDay}`, opts);
  return r.json();
};


// ---- per-day memo (RAM + sessionStorage) for /summary ----
const SUMMARY_CACHE = new Map();
const ssKey = (d) => `ss:summary:${d}`;

async function getSummaryCached(isoUtcDay, opts = {}) {
  // RAM cache
  if (SUMMARY_CACHE.has(isoUtcDay)) return SUMMARY_CACHE.get(isoUtcDay);

  // sessionStorage cache
  const hit = sessionStorage.getItem(ssKey(isoUtcDay));
  if (hit) {
    const parsed = JSON.parse(hit);
    SUMMARY_CACHE.set(isoUtcDay, parsed);
    return parsed;
  }

  // fetch and store
  const res = await warmFetch(`${API_BASE}/summary/${isoUtcDay}`, opts);
  const json = await res.json();
  SUMMARY_CACHE.set(isoUtcDay, json);
  sessionStorage.setItem(ssKey(isoUtcDay), JSON.stringify(json));
  return json;
}



// ---- hourly → chart rows (24 points) ----
function hourlyToSeries(isoDay, hourlyPred = [], hourlyAct = []) {
  const base = new Date(`${isoDay}T00:00:00Z`).getTime();
  const p = new Map(hourlyPred.map(h => [h.hour, Number(h.long_flux_pred)]));
  const a = new Map(hourlyAct .map(h => [h.hour, Number(h.long_flux     )]));
  const rows = [];
  for (let h = 0; h < 24; h++) {
    rows.push({
      t: new Date(base + h * 3600_000),
      pred: p.get(h) ?? null,
      actual: a.get(h) ?? null,
    });
  }
  return rows;
}

// ---- utilities for peaks from hourly ----
const hh = (n) => String(n).padStart(2, "0");
function peakFromHourly(hourly, fluxKey) {
  if (!Array.isArray(hourly) || hourly.length === 0) return null;
  let best = null;
  for (const h of hourly) {
    const v = Number(h[fluxKey]);
    if (!Number.isFinite(v)) continue;
    if (!best || v > best[fluxKey]) best = h;
  }
  if (!best) return null;
  return {
    class: `${best.class}-Class`,     // server already sends class for hourly rows
    utc: `${hh(best.hour)}:00`,
    flux: Number(best[fluxKey]),     
  };
}



/* Fallback: pull peaks from a summary sentence */
const parsePeaksFromSummary = (summary = "") => {
  const pred = summary.match(/Predicted\s+peak\s+([ABCMX])\s+at\s+~?(\d{2}:\d{2})/i);
  const obs  = summary.match(/Observed\s+peak\s+([ABCMX])\s+at\s+~?(\d{2}:\d{2})/i);
  return {
    pred_peak: pred ? { class: `${pred[1]}-Class`, utc: pred[2] } : null,
    obs_peak:  obs  ? { class: `${obs[1]}-Class`,  utc: obs[2] }  : null,
  };
};

const fmtFlux = (v) => new Intl.NumberFormat("en", { notation: "scientific", maximumFractionDigits: 2 }).format(v || 0);

const hoursToChart = (hourlyPred = [], hourlyAct = [], dayIso) => {
  // build one point per hour on the requested UTC day
  const mkTs = (h) => new Date(`${dayIso}T${String(h).padStart(2,"0")}:00:00Z`);
  const byHour = new Map();
  for (const h of hourlyPred) byHour.set(h.hour, { t: mkTs(h.hour), pred: Number(h.long_flux_pred) || undefined });
  for (const h of hourlyAct) {
    const row = byHour.get(h.hour) || { t: mkTs(h.hour) };
    row.actual = Number(h.long_flux) || undefined;
    byHour.set(h.hour, row);
  }
  return [...byHour.values()].sort((a,b)=>a.t-b.t);
};

// const peakFromHourly = (rows, fluxKey = "long_flux", classKey = "class") => {
//   if (!rows?.length) return null;
//   const best = rows.reduce((a,b) => ( (b[fluxKey]||0) > (a[fluxKey]||0) ? b : a ));
//   return {
//     class: `${best[classKey]}-Class`,
//     utc: `${String(best.hour).padStart(2,"0")}:00`,
//     flux: best[fluxKey] || 0,
//   };
// };

// === Quick “Pred vs Observed” quiz ===
// function PeakQuiz({ pred, obs, dark }) {
//   // Guard against missing/zero flux (supports .flux or .value)
//   const safe = (x) => (Number.isFinite(x) && x > 0 ? x : null);
//   const p = safe(pred?.flux ?? pred?.value);
//   const o = safe(obs?.flux ?? obs?.value);

//   if (!p || !o) {
//     return (
//       <div className={`mt-2 rounded-md p-2.5 ${dark ? "bg-slate-900/50" : "bg-slate-50"}`}>
//         <div className="text-xs opacity-70">Quick check</div>
//         <div className="mt-1 text-xs">Need both flux values to compare.</div>
//       </div>
//     );
//   }

//   const ratio = o / p;               // >1 → observed > predicted (model LOW)
//   const absErr = Math.abs(ratio - 1);
//   const within = 0.2;                // 20% window counts as “About right”
//   const truth = absErr <= within ? "right" : ratio > 1 ? "low" : "high";
//   const [answer, setAnswer] = React.useState(null);

//   const classOrder = (c) => {
//     if (!c) return -1;
//     const L = String(c).trim().toUpperCase()[0];
//     return L === "A" ? 0 : L === "B" ? 1 : L === "C" ? 2 : L === "M" ? 3 : L === "X" ? 4 : -1;
//   };
//   const deltaClass = classOrder(obs?.class) - classOrder(pred?.class);

//   const pct = (n) => `${(n * 100).toFixed(0)}%`;
//   const xFmt = (x) => {
//     const s = (x >= 10 ? x.toFixed(1) : x.toFixed(2)).replace(/\.0+$/, "");
//     return `${s}×`;
//   };

//   const bg = dark ? "bg-slate-900/50" : "bg-slate-50";
//   const btnBase = "px-2.5 py-1.5 text-xs rounded-md border transition-colors";
//   const btnLight = dark ? "border-slate-700 hover:bg-slate-800" : "border-slate-300 hover:bg-slate-100";
//   const btnSelected = "border-transparent ring-1 ring-indigo-500/70";

//   const explanation =
//     truth === "right"
//       ? `Within ${pct(within)} ⇒ about right.`
//       : truth === "low"
//       ? "Observed > Predicted ⇒ model was LOW."
//       : "Observed < Predicted ⇒ model was HIGH.";

//   return (
//     <div className={`mt-2 ${bg} rounded-md p-2.5`}>
//       <div className="flex items-start justify-between gap-3">
//         <div className="text-xs opacity-70">Quick check</div>
//         <div className="flex gap-1.5">
//           {["high", "right", "low"].map((opt) => (
//             <button
//               key={opt}
//               className={`${btnBase} ${btnLight} ${answer === opt ? btnSelected : ""}`}
//               onClick={() => setAnswer(opt)}
//               title={
//                 opt === "high"
//                   ? "Model predicted too HIGH"
//                   : opt === "low"
//                   ? "Model predicted too LOW"
//                   : "Model was ABOUT RIGHT"
//               }
//             >
//               {opt === "right" ? "About right" : opt === "high" ? "High" : "Low"}
//             </button>
//           ))}
//         </div>
//       </div>

//       <div className="mt-2 text-xs leading-relaxed">
//         <div className="opacity-80">
//           Observed is <span className="font-medium">{xFmt(ratio)}</span> of predicted (
//           {ratio > 1 ? "+" : "−"}
//           {pct(Math.abs(ratio - 1))}).{" "}
//           <span className="opacity-70">
//             Classes: {pred?.class || "—"} → {obs?.class || "—"}{" "}
//             {Number.isFinite(deltaClass)
//               ? deltaClass === 0
//                 ? "(same letter class)"
//                 : `(${deltaClass > 0 ? "+" : ""}${deltaClass} class step${Math.abs(deltaClass) === 1 ? "" : "s"})`
//               : ""}
//           </span>
//         </div>

//         {answer && (
//           <div
//             className={`mt-2 rounded-md px-2 py-1 ${
//               answer === truth
//                 ? dark
//                   ? "bg-emerald-900/40 text-emerald-300"
//                   : "bg-emerald-50 text-emerald-700"
//                 : dark
//                 ? "bg-rose-900/40 text-rose-300"
//                 : "bg-rose-50 text-rose-700"
//             }`}
//           >
//             {answer === truth ? "Correct." : "Incorrect."} {explanation}
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }



/* ---------- main ---------- */

const App = () => {
  const [day, setDay] = useState(todayUTC());
  const [flareData, setData] = useState(makeDummy()); // hourly tiles
  const [chartData, setChartData] = useState([]);     // minute flux
  const [dark, setDark] = useState(true);
  const [now, setNow] = useState(new Date());
  const [sdo, setSdo] = useState(null);
  const [fadeKey, setFadeKey] = useState(0);          // for SDO panel fade

  // const forecastAbortRef = useRef(null);
  // const summaryAbortRef  = useRef(null);
  const sdoAbortRef      = useRef(null);

  // limit how far back a user can go (in UTC)
  const MIN_DAYS_BACK = 6;
  const minAllowedUTC = shiftDay(todayUTC(), -MIN_DAYS_BACK);

  const isToday  = day === todayUTC();
  const isAtMin  = day <= minAllowedUTC; // "YYYY-MM-DD" compares lexicographically

  // video + sizing refs
  const videoRef = useRef(null);
  const timeRef  = useRef(null);
  const leftColRef = useRef(null);
  const [vidH, setVidH] = useState(420);              // desktop video height

  const [peaks, setPeaks] = useState({ pred_peak: null, obs_peak: null });

  const [movieUrl, setMovieUrl] = useState(null); // currently selected video URL
  const [movieFallbackTry, setMovieFallbackTry] = useState(0); // 0=none,1=yesterday,2=latest
  const [movieTryIso, setMovieTryIso] = useState(null); // which day the player is trying
  const [movieTries, setMovieTries] = useState(0);      // 0 = day, 1 = day-1 (stop after)
  const [similarity, setSimilarity] = useState({ same: 0, diff: 0, total: 0 });



  // prevent moving into the future (UTC)
  const goPrev = () =>
    setDay((d) => {
      const min = shiftDay(todayUTC(), -MIN_DAYS_BACK);
      const n   = shiftDay(d, -1);
      return n < min ? d : n;  // no-op if we’d go past the limit
    });
  const goNext = () =>
    setDay((d) => {
      const t = todayUTC();
      const n = shiftDay(d, 1);
      return n > t ? d : n;
    });

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const s = await getSummary(day);
        if (cancelled || !s?.hourly_pred?.length) return;

        // Hourly tiles
        const predHours = s.hourly_pred.map(h => ({
          hour: h.hour,
          time: `${h.hour % 12 || 12}${h.hour < 12 ? "AM" : "PM"}`,
          predClass: `${h.class}-Class`,
        }));
        const actualByHour = new Map((s.hourly_actual || []).map(h => [h.hour, `${h.class}-Class`]));
        const hours = predHours.map(h => ({
          ...h,
          actualClass: actualByHour.get(h.hour) || null,
          classIndex: ["A","B","C","M","X"].indexOf((h.predClass||"A").charAt(0)),
        }));
        setData(hours);

        // Chart from hourly (fast)
        setChartData(hoursToChart(s.hourly_pred, s.hourly_actual, s.date));

        // Peaks for the side cards (always available)
        const predP = peakFromHourly(s.hourly_pred, "long_flux_pred", "class");
        const obsP  = peakFromHourly(s.hourly_actual, "long_flux",      "class");
        setPeaks({ pred_peak: predP, obs_peak: obsP });

        // similarity stats: compare per-hour predicted vs actual class
        let same = 0, diff = 0;
        if (s.hourly_pred && s.hourly_actual) {
          const actByHour = new Map(s.hourly_actual.map(h => [h.hour, h.class]));
          for (const h of s.hourly_pred) {
            const act = actByHour.get(h.hour);
            if (!act) continue;
            if (String(h.class).toUpperCase()[0] === String(act).toUpperCase()[0]) same++;
            else diff++;
          }
        }
        setSimilarity({ same, diff, total: same + diff });


      } catch (e) {
        console.warn("summary load failed:", e);
        setChartData([]);
        setData(makeDummy());
        setPeaks({ pred_peak: null, obs_peak: null });
      }
    })();

    return () => { cancelled = true; };
  }, [day]);



  useEffect(() => {
    let cancelled = false;
    const ac = new AbortController();

    (async () => {
      try {
        const payload = await getSdoLite(day, { signal: ac.signal });
        if (cancelled) return;
        setSdo(payload || null);

        // always try the date-specific GSFC daily first
        setMovieUrl(gsfcAIA171UrlFor(day));
        setMovieTryIso(day);
        setMovieTries(0);

        setFadeKey(k => k + 1);
      } catch (e) {
        if (e?.name !== "AbortError") console.warn("sdo-lite load failed:", e);
        setSdo(null);
      }
    })();

    return () => { cancelled = true; ac.abort(); };
  }, [day]);


  // live clock
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);

  // measure left column height → set video height (desktop)
  const measureHeights = () => {
    if (!leftColRef.current) return;
    const h = Math.max(360, leftColRef.current.offsetHeight);
    setVidH(h);
  };
  useEffect(() => {
    measureHeights();
  }, [sdo, dark]);
  useEffect(() => {
    const onResize = () => measureHeights();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const currentHour = now.getUTCHours();
  const currentTime = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });

  // theme palette
  const bgGrad = dark ? "from-slate-800 to-slate-900" : "from-sky-100 to-sky-300";
  const panel = dark ? "bg-slate-700" : "bg-white/40 backdrop-blur";
  const textMain = dark ? "text-white" : "text-slate-800";
  const textSub = dark ? "text-slate-300" : "text-slate-600";
  const arrowClr = dark ? "text-blue-300" : "text-blue-700";
  const hiBg = dark ? "bg-cyan-500 text-white" : "bg-blue-600 text-white";
  const maskColor = dark ? "rgba(30,41,59,0.88)" : "rgba(226,232,240,0.90)";

  // y-axis bounds
  const allVals = chartData.flatMap((d) =>
    [d.pred, d.actual].filter((x) => x && x > 0)
  );
  const dMin = allVals.length ? Math.min(...allVals) : 1e-7;
  const dMax = allVals.length ? Math.max(...allVals) : 1e-5;
  const yMin = Math.max(1e-8, dMin * 0.8);
  const yMax = Math.min(1e-3, dMax * 1.2);

  // headline class
  const headLevel = flareData?.[0]?.predClass || flareData?.[0]?.level || "";

  // parse peaks if needed
  // const peaks = sdo
  //   ? {
  //       ...(sdo.pred_peak && { pred_peak: sdo.pred_peak }),
  //       ...(sdo.obs_peak && { obs_peak: sdo.obs_peak }),
  //       ...(!sdo.pred_peak || !sdo.obs_peak
  //         ? parsePeaksFromSummary(sdo.summary)
  //         : {}),
  //     }
  //   : {};


  // update tiny UTC time label w/o re-render
  const onVideoTimeUpdate = () => {
    const v = videoRef.current;
    const el = timeRef.current;
    if (!v || !el || !v.duration || !Number.isFinite(v.duration)) return;
    const frac = Math.max(0, Math.min(1, v.currentTime / v.duration));
    const totalMin = Math.round(frac * 24 * 60);
    const hh = String(Math.floor(totalMin / 60)).padStart(2, "0");
    const mm = String(totalMin % 60).padStart(2, "0");
    el.textContent = `${hh}:${mm} UTC`;
  };

  const onVideoError = () => {
    // If today’s daily movie isn’t published yet (usually ~21:00 UTC), fall back fast.
    if (movieTries === 0) {
      const prev = shiftDay(day, -1);
      setMovieUrl(gsfcAIA171UrlFor(prev));
      setMovieTryIso(prev);
      setMovieTries(1);
      return;
    }
    if (movieFallbackTry === 1) {
      setMovieUrl(SWPC_AIA171_LATEST_MP4);
      setMovieFallbackTry(2);
      return;
    }
    // Already tried all fallbacks — keep whatever is set.
  };


  // choose peak sources: hourly → SDO → parsed summary text
  const peakPred = peaks?.pred_peak || sdo?.pred_peak || null;
  const peakObs  = peaks?.obs_peak  || sdo?.obs_peak  || null;

  return (
    <div className={`bg-gradient-to-b ${bgGrad} ${textMain} min-h-screen p-4 sm:p-6 font-sans transition-colors duration-300`}>
      {/* title bar */}
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-3xl sm:text-4xl font-extrabold">Solar&nbsp;Sense</h1>
        <button onClick={() => setDark((x) => !x)}>
          <img
            src={
              dark
                ? "https://img.icons8.com/emoji/96/sun-emoji.png"
                : "https://img.icons8.com/emoji/96/crescent-moon-emoji.png"
            }
            alt="mode"
            className="w-10 h-10"
          />
        </button>
      </header>

      {/* date + headline */}
      <section className="flex items-center justify-center gap-4 mb-6">
        <button
          onClick={goPrev}
          disabled={isAtMin}
          className={`px-3 text-2xl font-bold select-none ${arrowClr} ${
            isAtMin ? "opacity-0 pointer-events-none" : ""
          }`}
        >
          &lt;
        </button>

        <div className="text-center">
          <p className={`text-lg font-semibold ${textSub}`}>{longDate(day)}</p>
          <p className={`text-sm mb-1 ${textSub}`}>{currentTime} UTC</p>
          <p className="text-3xl font-bold">{headLevel}</p>
        </div>
        <button
          onClick={goNext}
          disabled={isToday}
          className={`px-3 text-2xl font-bold select-none ${arrowClr} ${isToday ? "opacity-0 pointer-events-none" : ""}`}
        >
          &gt;
        </button>
      </section>

      {/* hourly strip */}
      <section className="mb-6">
        <div className={`hidden sm:flex overflow-x-auto gap-3 ${panel} rounded-xl p-3`}>
          {flareData.map((h, i) => {
            const hi = i === currentHour ? hiBg : "";
            return (
              <div key={i} className="text-center min-w-[64px]">
                <p className={`text-sm font-semibold rounded ${hi}`}>{h.time}</p>
                <p className="text-xs text-sky-300">{h.predClass || h.level}</p>
                {h.actualClass && (
                  <p className="text-xs text-emerald-300">{h.actualClass}</p>
                )}
              </div>
            );
          })}
        </div>
        <div className={`sm:hidden overflow-y-auto max-h-64 flex flex-col gap-2 ${panel} rounded-xl p-3`}>
          {flareData.map((h, i) => {
            const hi = i === currentHour ? hiBg : "bg-white/20 sm:bg-transparent";
            return (
              <div key={i} className={`rounded-lg px-3 py-1 ${hi}`}>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">{h.time}</span>
                  <span className="text-xs text-sky-300">{h.predClass || h.level}</span>
                </div>
                {h.actualClass && (
                  <div className="text-right text-xs text-emerald-300">{h.actualClass}</div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* chart */}
      <section className={`${panel} p-4 rounded-xl mb-6 overflow-hidden relative`}>
        <h3 className={`flex items-center justify-center text-lg mb-2 ${textMain}`}>Minute-Level Flux (Prediction vs Actual)</h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData}>
            <XAxis
              dataKey="t"
              tickFormatter={(d) => {
                const s = typeof d === "string" ? d : new Date(d).toISOString();
                const dt = new Date(s.endsWith("Z") ? s : s + "Z");
                return dt.toLocaleTimeString([], { hour: "2-digit", hour12: true, timeZone: "UTC" });
              }}
              minTickGap={30}
              stroke={dark ? "#94a3b8" : "#334155"}
            />
            <YAxis
              scale="log"
              domain={[yMin, yMax]}
              tick={{ fill: dark ? "#94a3b8" : "#334155", fontSize: 11 }}
              tickFormatter={(v) => sci.format(v)}
              width={60}
            />
            <Tooltip
              wrapperStyle={{ pointerEvents: "none" }}
              contentStyle={{ background: dark ? "#1e293b" : "#f1f5f9", border: "none" }}
              labelFormatter={(d) => {
                const s = typeof d === "string" ? d : new Date(d).toISOString();
                const dt = new Date(s.endsWith("Z") ? s : s + "Z");
                return dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
              }}
              formatter={(val, name) => [sci.format(val), name]}
            />
            <Legend />
            <Line type="monotone" dataKey="actual" name="Actual Flux"    stroke="#10b981" strokeWidth={1.5} dot={false} connectNulls isAnimationActive animationDuration={700} />
            <Line type="monotone" dataKey="pred"   name="Predicted Flux" stroke="#38bdf8" strokeWidth={1.5} dot={false} connectNulls isAnimationActive animationDuration={700} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* 171 Å Solar View */}
      <section key={fadeKey} className={`${panel} p-4 rounded-xl mb-6 transition-opacity duration-500 opacity-100`}>
        <h3 className={`flex items-center justify-center text-lg mb-3 ${textMain}`}>171 Å Solar View</h3>

        <div className="grid md:grid-cols-3 gap-4 items-stretch">
          {/* LEFT: cards (natural height; drives video height) */}
          <div ref={leftColRef} className="md:col-span-1 space-y-3">
            <div className={`p-3 rounded-lg ${dark ? "bg-slate-800/60" : "bg-white/70"}`}>
              <p className="text-xs opacity-70">Predicted Peak</p>
              {peaks?.pred_peak && (
                <p className="text-sm">
                  Predicted {peaks.pred_peak.class} at {peaks.pred_peak.utc} UTC, with a flux value of ({fmtFlux(peaks.pred_peak.flux)})
                </p>
              )}
            </div>

            <div className={`p-3 rounded-lg ${dark ? "bg-slate-800/60" : "bg-white/70"}`}>
              <p className="text-xs opacity-70">Observed Peak</p>
              {peaks?.obs_peak && (
                <p className="text-sm">
                  Actual {peaks.obs_peak.class} at {peaks.obs_peak.utc} UTC, with a flux value of ({fmtFlux(peaks.obs_peak.flux)})
                </p>
              )}
            </div>

            <div className={`p-3 rounded-lg ${dark ? "bg-slate-800/60" : "bg-white/70"}`}>
              <p className="text-xs opacity-70 mb-1">Summary</p>

              {peaks?.pred_peak && (
                <p className="text-sm">
                  Predicted peak: {peaks.pred_peak.class} at {peaks.pred_peak.utc} UTC ({fmtFlux(peaks.pred_peak.flux)})
                </p>
              )}

              {peaks?.obs_peak && (
                <p className="text-sm">
                  Observed peak: {peaks.obs_peak.class} at {peaks.obs_peak.utc} UTC ({fmtFlux(peaks.obs_peak.flux)})
                </p>
              )}

              {peaks?.pred_peak && peaks?.obs_peak && (
                <p className="text-sm">
                  Difference: {fmtFlux(peaks.obs_peak.flux - peaks.pred_peak.flux)}
                </p>
              )}

              <p className="text-sm">
                  ---------- 
              </p>
              {/* similarity stats */}
              {similarity.total > 0 && (
                <p className="text-sm mt-1">
                  Hourly match: {similarity.same}/{similarity.total} (
                  {((similarity.same / similarity.total) * 100).toFixed(0)}% similarity)
                </p>
              )}

              <p className="text-sm">
                  ---------- 
              </p>

              {/* keep the plain descriptive quick-check sentence */}
              {peaks?.pred_peak && peaks?.obs_peak && (
                <p className="text-xs mt-1">
                  Observed is{" "}
                    {(peaks.obs_peak.flux / peaks.pred_peak.flux).toFixed(2)}×
                  of predicted (
                  {((peaks.obs_peak.flux / peaks.pred_peak.flux - 1) * 100).toFixed(0)}%). Classes:{" "}
                  {peaks.pred_peak.class} - {peaks.obs_peak.class}
                </p>
              )}

              {/* {(() => {
              const peakPred = peaks?.pred_peak || sdo?.pred_peak || null;
              const peakObs  = peaks?.obs_peak  || sdo?.obs_peak  || null;
              return peakPred && peakObs ? (
                <PeakQuiz pred={peakPred} obs={peakObs} dark={dark} />
              ) : null;
              })()} */}

            </div>

            {!!(sdo?.regions?.length) && (
              <div className={`p-3 rounded-lg ${dark ? "bg-slate-800/60" : "bg-white/70"}`}>
                <p className="text-xs opacity-70 mb-2">Highlighted bright regions</p>
                <ul className="text-sm space-y-1">
                  {sdo.regions.map((r, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-500 text-white text-[10px]">{i + 1}</span>
                      <span>score {Math.round(r.score)}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-[11px] opacity-70">
                  Scores are a relative brightness metric (higher = brighter region).
                </p>
              </div>
            )}
          </div>

          {/* RIGHT: stretched video sized to match left height on desktop */}
          <div className="md:col-span-2 relative rounded-lg overflow-hidden w-full md:block">
            <div className="relative w-full md:w-full"
                 style={{ height: `${vidH}px` }}>
              {sdo?.movie_url ? (
                <>
                  <video
                    ref={videoRef}
                    key={movieUrl || "no-video"}
                    src={movieUrl || ""}
                    poster={sdo?.poster_url || ""}  // fine to keep; or omit
                    preload="metadata"
                    autoPlay
                    loop
                    muted
                    playsInline
                    onTimeUpdate={onVideoTimeUpdate}
                    onError={onVideoError}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      objectPosition: "center 40%", // nudged up a bit
                      transform: "scale(1.02)",
                    }}
                  />
                  {/* masks to fully hide SDO caption */}
                  <div
                    className="absolute inset-x-0 bottom-0 pointer-events-none"
                    style={{ height: "26%", background: `linear-gradient(to bottom, transparent, ${maskColor})` }}
                  />
                  <div
                    className="absolute inset-x-0 bottom-0 pointer-events-none"
                    style={{ height: "12%", background: maskColor }}
                  />
                  {/* region overlays */}
                  {(sdo.regions || []).map((r, idx) => {
                    const toPct = (v) => (v <= 1 ? v * 100 : (v / 1024) * 100);
                    const xPct = Math.max(0, Math.min(100, toPct(r.x)));
                    const yPct = Math.max(0, Math.min(100, toPct(r.y)));
                    const dPct = Math.max(0, Math.min(200, toPct(r.r) * 2));
                    return (
                      <div key={idx}>
                        <div
                          title={`Region ${idx + 1} • score ${Math.round(r.score)}`}
                          style={{
                            position: "absolute",
                            left: `${xPct}%`,
                            top: `${yPct}%`,
                            width: `${dPct}%`,
                            height: `${dPct}%`,
                            transform: "translate(-50%, -50%)",
                            border: "2px solid rgba(99,102,241,0.9)",
                            borderRadius: "9999px",
                            boxShadow: "0 0 12px rgba(99,102,241,0.8)",
                            pointerEvents: "none",
                          }}
                        />
                        <span
                          style={{
                            position: "absolute",
                            left: `${xPct}%`,
                            top: `calc(${yPct}% - 1.2rem)`,
                            transform: "translate(-50%, -50%)",
                            background: "rgba(99,102,241,0.95)",
                            color: "white",
                            padding: "2px 6px",
                            borderRadius: "9999px",
                            fontSize: 10,
                            pointerEvents: "none",
                          }}
                        >
                          {idx + 1}
                        </span>
                      </div>
                    );
                  })}
                  {/* tiny UTC time label */}
                  <div
                    ref={timeRef}
                    className="absolute left-2 bottom-2 text-[10px] px-2 py-[2px] rounded bg-black/40 text-white"
                  >
                    00:00 UTC
                  </div>
                </>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-sm">
                  No video available for this date.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <footer className={`text-center text-xs ${textSub}`}>
        Updated {now.toLocaleTimeString()} · UTC Timezone
      </footer>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);