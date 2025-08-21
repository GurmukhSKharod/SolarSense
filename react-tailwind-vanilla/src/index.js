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
const API_BASE =
  // Vite-style override if present
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE) ||
  // window override if you ever want to set it in index.html:
  (typeof window !== "undefined" && window.__API_BASE__) ||
  // local dev fallback
  ((location.hostname === "localhost" || location.hostname === "127.0.0.1")
    ? "http://localhost:8000"
    // production: use path that can be proxied by Netlify
    : "/api");


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

const getForecast = async (isoLocalDay) => {
  try {
    const dateUTC = localDayToUTCISO(isoLocalDay);
    const r = await fetch(`${API_BASE}/forecast/${dateUTC}`, { mode: "cors" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.error("getForecast failed:", e, "API_BASE:", API_BASE);
    return null;
  }
};

const getSdo = async (isoLocalDay) => {
  try {
    const dateUTC = localDayToUTCISO(isoLocalDay);
    const r = await fetch(`${API_BASE}/sdo/${dateUTC}`, { mode: "cors" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.error("getSdo failed:", e, "API_BASE:", API_BASE);
    return null;
  }
};

/* Fallback: pull peaks from a summary sentence */
const parsePeaksFromSummary = (summary = "") => {
  const pred = summary.match(/Predicted\s+peak\s+([ABCMX])\s+at\s+~?(\d{2}:\d{2})/i);
  const obs  = summary.match(/Observed\s+peak\s+([ABCMX])\s+at\s+~?(\d{2}:\d{2})/i);
  return {
    pred_peak: pred ? { class: `${pred[1]}-Class`, utc: pred[2] } : null,
    obs_peak:  obs  ? { class: `${obs[1]}-Class`,  utc: obs[2] }  : null,
  };
};

/* ---------- main ---------- */

const App = () => {
  const [day, setDay] = useState(todayUTC());
  const [flareData, setData] = useState(makeDummy()); // hourly tiles
  const [chartData, setChartData] = useState([]);     // minute flux
  const [dark, setDark] = useState(true);
  const [now, setNow] = useState(new Date());
  const [sdo, setSdo] = useState(null);
  const [fadeKey, setFadeKey] = useState(0);          // for SDO panel fade

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
    (async () => {
      const data = await getForecast(day);

      // ----- Hourly strip (pred + actual when available)
      if (data && data.hourly_pred?.length === 24) {
        const predHours = data.hourly_pred.map((h) => ({
          hour: h.hour,
          time: `${h.hour % 12 || 12}${h.hour < 12 ? "AM" : "PM"}`,
          predClass: `${h.class}-Class`,
        }));
        const actualByHour = new Map(
          (data.hourly_actual || []).map((h) => [h.hour, `${h.class}-Class`])
        );
        const hours = predHours.map((h) => ({
          ...h,
          actualClass: actualByHour.get(h.hour) || null,
          classIndex: ["A", "B", "C", "M", "X"].indexOf(
            (h.predClass || "A").charAt(0)
          ),
        }));
        setData(hours);
      } else {
        setData(makeDummy());
      }

      // ----- Minute series (Pred + Actual)
      const pred = (data?.minute_pred || [])
        .map((d) => [Date.parse(d.timestamp), Number(d.long_flux_pred)])
        .filter(([, v]) => Number.isFinite(v) && v > 0);

      const act = (data?.minute_actual || [])
        .map((d) => [Date.parse(d.timestamp), Number(d.long_flux)])
        .filter(([, v]) => Number.isFinite(v) && v > 0);

      if (pred.length || act.length) {
        const map = new Map();
        for (const [ts, v] of pred) map.set(ts, { t: new Date(ts), pred: v });
        for (const [ts, v] of act) {
          const row = map.get(ts) || { t: new Date(ts) };
          row.actual = v;
          map.set(ts, row);
        }
        const merged = [...map.entries()]
          .sort((a, b) => a[0] - b[0])
          .map(([, v]) => v);
        setChartData(merged);
      } else {
        setChartData([]);
      }
    })();
  }, [day]);

  // SDO payload + fade
  useEffect(() => {
    (async () => {
      const payload = await getSdo(day);
      setSdo(payload || null);
      setFadeKey((k) => k + 1);
    })();
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
  const peaks = sdo
    ? {
        ...(sdo.pred_peak && { pred_peak: sdo.pred_peak }),
        ...(sdo.obs_peak && { obs_peak: sdo.obs_peak }),
        ...(!sdo.pred_peak || !sdo.obs_peak
          ? parsePeaksFromSummary(sdo.summary)
          : {}),
      }
    : {};

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
              <p className="text-sm font-semibold">
                {peaks?.pred_peak ? `${peaks.pred_peak.class} at ${peaks.pred_peak.utc} UTC` : "—"}
              </p>
            </div>

            <div className={`p-3 rounded-lg ${dark ? "bg-slate-800/60" : "bg-white/70"}`}>
              <p className="text-xs opacity-70">Observed Peak</p>
              <p className="text-sm font-semibold">
                {peaks?.obs_peak ? `${peaks.obs_peak.class} at ${peaks.obs_peak.utc} UTC` : "—"}
              </p>
            </div>

            <div className={`p-3 rounded-lg ${dark ? "bg-slate-800/60" : "bg-white/70"}`}>
              <p className="text-xs opacity-70 mb-1">Summary</p>
              <p className="text-sm leading-relaxed">{sdo?.summary || "—"}</p>
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
                    key={sdo.movie_url}
                    src={sdo.movie_url}
                    autoPlay
                    loop
                    muted
                    playsInline
                    onTimeUpdate={onVideoTimeUpdate}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",     // stretch & crop
                      objectPosition: "center 46%",
                      transform: "scale(1.03)", // tiny zoom to bury any caption edges
                    }}
                  />
                  {/* masks to fully hide SDO caption */}
                  <div
                    className="absolute inset-x-0 bottom-0 pointer-events-none"
                    style={{ height: "22%", background: `linear-gradient(to bottom, transparent, ${maskColor})` }}
                  />
                  <div
                    className="absolute inset-x-0 bottom-0 pointer-events-none"
                    style={{ height: "10%", background: maskColor }}
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
                  No imagery available for this date.
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