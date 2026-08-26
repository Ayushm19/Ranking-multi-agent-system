/* Ranking Agent workbench — upload resumes + paste JD, results below. */

(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const jd = $('jd');
  const resultsEl = $('results');
  const emptyEl = $('empty');
  const statusSlot = $('status-slot');
  const outputRegion = $('output-region');
  const runBtn = $('run');

  let pendingFiles = [];

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  };

  const icon = (path, size = 15) => {
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('aria-hidden', 'true');
    const p = document.createElementNS(NS, 'path');
    p.setAttribute('d', path);
    svg.appendChild(p);
    return svg;
  };

  const ICONS = {
    warn: 'M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z',
    check: 'M20 6 9 17l-5-5',
    info: 'M12 16v-4M12 8h.01M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z',
    shield: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
    chevron: 'M6 9l6 6 6-6',
    x: 'M18 6 6 18M6 6l12 12',
  };

  function updateCounts() {
    const n = pendingFiles.length;
    $('cand-count').textContent = n
      ? `${n} file${n === 1 ? '' : 's'}`
      : 'none yet';
    $('jd-count').textContent = `${jd.value.length} chars`;
    renderFileChips();
  }

  function setPendingFiles(files, { replace = false } = {}) {
    const next = Array.from(files || []);
    if (replace) {
      pendingFiles = next;
    } else {
      const map = new Map(pendingFiles.map((f) => [`${f.name}:${f.size}`, f]));
      for (const f of next) map.set(`${f.name}:${f.size}`, f);
      pendingFiles = Array.from(map.values());
    }
    updateCounts();
    if (pendingFiles.length) {
      setStatus('ok', `${pendingFiles.length} file(s) ready to rank.`);
    }
  }

  function removePendingFile(key) {
    pendingFiles = pendingFiles.filter((f) => `${f.name}:${f.size}` !== key);
    updateCounts();
  }

  function renderFileChips() {
    const host = $('file-chips');
    if (!host) return;
    host.replaceChildren();
    if (!pendingFiles.length) {
      host.hidden = true;
      return;
    }
    host.hidden = false;
    for (const f of pendingFiles) {
      const key = `${f.name}:${f.size}`;
      const chip = el(
        'div',
        'inline-flex max-w-full items-center gap-2 rounded-md border border-white/10 bg-[#0c0c0e] px-2.5 py-1.5 text-[11px] font-mono text-zinc-300',
      );
      const name = el('span', 'truncate');
      name.textContent = f.name;
      const meta = el('span', 'text-zinc-600 shrink-0', `${Math.max(1, Math.round(f.size / 1024))}kb`);
      const btn = el(
        'button',
        'inline-flex h-5 w-5 items-center justify-center rounded-md bg-white/5 text-zinc-400 hover:text-red-400 transition',
      );
      btn.type = 'button';
      btn.setAttribute('aria-label', `Remove ${f.name}`);
      btn.appendChild(icon(ICONS.x, 10));
      btn.addEventListener('click', () => removePendingFile(key));
      chip.append(name, meta, btn);
      host.appendChild(chip);
    }
  }

  function wireDropzone() {
    const zone = $('dropzone');
    const input = $('files');
    if (!zone || !input) return;

    const openPicker = () => input.click();
    zone.addEventListener('click', openPicker);
    zone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openPicker();
      }
    });

    input.addEventListener('change', (e) => {
      setPendingFiles(e.target.files);
      input.value = '';
    });

    for (const evt of ['dragenter', 'dragover']) {
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        zone.classList.add('is-drag');
      });
    }
    zone.addEventListener('dragleave', () => {
      zone.classList.remove('is-drag');
    });
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('is-drag');
      if (e.dataTransfer && e.dataTransfer.files.length) {
        setPendingFiles(e.dataTransfer.files);
      }
    });
  }

  function setStatus(kind, message, busy = false) {
    statusSlot.replaceChildren();
    if (!message) {
      outputRegion.setAttribute('aria-busy', 'false');
      return;
    }
    let cls =
      'mb-3 flex items-center gap-3 rounded-lg border px-3.5 py-2.5 text-sm ';
    if (kind === 'err') {
      cls += 'border-red-500/35 bg-red-500/[0.08] text-red-100';
    } else if (kind === 'ok') {
      cls += 'border-emerald-500/25 bg-emerald-500/[0.08] text-emerald-100';
    } else {
      cls += 'border-white/10 bg-[#111113] text-zinc-300';
    }
    const box = el('div', cls);
    box.setAttribute('data-testid', 'status');
    if (kind) box.setAttribute('data-status', kind);
    if (busy) {
      const spin = el(
        'span',
        'h-4 w-4 shrink-0 rounded-full border-2 border-zinc-600 border-t-emerald-400 animate-spin',
      );
      box.appendChild(spin);
    } else {
      box.appendChild(icon(kind === 'err' ? ICONS.warn : ICONS.check, 16));
    }
    box.appendChild(el('span', null, message));
    statusSlot.appendChild(box);
    outputRegion.setAttribute('aria-busy', busy ? 'true' : 'false');
  }

  const scoreBar = (v) =>
    v >= 70 ? 'bg-[#3ecf8e]' : v >= 45 ? 'bg-amber-400' : 'bg-red-400';

  const scoreTone = (v) =>
    v >= 70 ? 'text-[#3ecf8e]' : v >= 45 ? 'text-amber-400' : 'text-red-400';

  const GLYPH = { exact: '=', transferable: '~', partial: '±', gap: '×' };
  const CHIP = {
    exact: 'border-[#3ecf8e]/35 text-[#3ecf8e] bg-[#3ecf8e]/10',
    transferable: 'border-sky-500/35 text-sky-300 bg-sky-500/10',
    partial: 'border-amber-500/35 text-amber-300 bg-amber-500/10',
    gap: 'border-red-500/35 text-red-300 bg-red-500/10',
  };

  function scoreDial(score) {
    const wrap = el('div', 'relative h-14 w-14 shrink-0');
    const circ = 2 * Math.PI * 15;
    const offset = circ * (1 - Math.max(0, Math.min(100, score)) / 100);
    const stroke =
      score >= 70 ? '#3ecf8e' : score >= 45 ? '#fbbf24' : '#f87171';
    wrap.innerHTML = `
      <svg viewBox="0 0 36 36" class="h-14 w-14 -rotate-90" aria-hidden="true">
        <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="3"/>
        <circle cx="18" cy="18" r="15" fill="none" stroke="${stroke}" stroke-width="3"
          stroke-linecap="round" stroke-dasharray="${circ.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"/>
      </svg>`;
    const label = el(
      'span',
      `absolute inset-0 grid place-items-center font-sans text-sm font-semibold tabular-nums ${scoreTone(score)}`,
      String(score),
    );
    label.setAttribute('data-testid', 'result-score-value');
    wrap.appendChild(label);
    return wrap;
  }

  function renderSummary(session) {
    const box = el('div', 'grid grid-cols-2 sm:grid-cols-4 gap-3');
    box.setAttribute('data-testid', 'summary');
    const reviewed = session.results.filter((r) => r.needs_human_review).length;
    const items = [
      ['ranked', `${session.results.length}/${session.total_candidates}`],
      ['top score', session.results.length ? session.results[0].final_score : '—'],
      ['review', reviewed],
      ['cost', `$${(session.total_cost_usd || 0).toFixed(4)}`],
    ];
    for (const [k, v] of items) {
      const cell = el('div', 'rounded-lg border border-white/8 bg-[#111113] px-3.5 py-3');
      cell.setAttribute('data-testid', 'summary-box');
      cell.append(
        el('div', 'text-[10px] uppercase tracking-wider font-mono text-zinc-500', k),
        el('div', 'mt-1.5 font-sans text-lg font-semibold tabular-nums text-zinc-100', v),
      );
      box.appendChild(cell);
    }
    return box;
  }

  let rankingResults = [];

  const esc = (s) =>
    String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');

  const scoreBand = (v) => (v >= 70 ? 'high' : v >= 45 ? 'mid' : 'low');
  const mtClass = (type) =>
    ({
      exact: 'mt-exact',
      transferable: 'mt-transferable',
      partial: 'mt-partial',
      gap: 'mt-gap',
    })[String(type || '').toLowerCase()] || 'mt-partial';

  const miniBar = (score) => {
    const n = Math.max(0, Math.min(100, Number(score) || 0));
    const color = n >= 70 ? '#34d399' : n >= 40 ? '#fbbf24' : '#f87171';
    return `<span class="mini-bar-track"><span class="mini-bar-fill" style="width:${n}%;background:${color}"></span></span>`;
  };

  function matchTableHtml(details, colName) {
    if (!details || !details.length) {
      return '<div class="match-empty">No detailed breakdown available.</div>';
    }
    return `<div class="match-list" data-testid="match-table">
      <div class="match-list-head" aria-hidden="true">
        <span>${esc(colName || 'Item')}</span>
        <span>Match</span>
        <span>Score</span>
      </div>
      ${details
        .map(
          (d) => `<div class="match-row" data-testid="match-row">
          <div class="match-row-main">
            <div class="match-row-item">${esc(d.item)}</div>
            <span class="match-type-badge ${mtClass(d.match_type)}" data-testid="match-badge">${esc(d.match_type)}</span>
            <div class="match-row-score">${Number(d.score || 0).toFixed(0)}${miniBar(d.score)}</div>
          </div>
          ${d.explanation ? `<div class="match-row-expl">${esc(d.explanation)}</div>` : ''}
        </div>`,
        )
        .join('')}
    </div>`;
  }

  function evidenceListHtml(quotes) {
    const items = (quotes || []).filter(Boolean);
    if (!items.length) return '';
    return `<ul class="evidence-list" data-testid="match-table">
      ${items
        .map((q) => {
          const text = typeof q === 'string' ? q : String(q);
          return `<li class="evidence-item" data-testid="match-row">${esc(text)}</li>`;
        })
        .join('')}
    </ul>`;
  }

  function buildDetailHtml(r) {
    let html = '';
    const v = r.verification || {};

    html += `<div class="detail-overview">
      <span class="ds-score ${scoreBand(r.final_score)}">${Number(r.final_score).toFixed(0)}/100</span>
      <span class="detail-overview-meta">${esc(
        [
          r.experience_level || null,
          r.candidate_years != null ? `${r.candidate_years} yrs` : null,
          `grounded ${Math.round((v.groundedness ?? 1) * 100)}%`,
          r.critic && r.critic.verdict ? `critic ${r.critic.verdict}` : null,
        ]
          .filter(Boolean)
          .join(' · '),
      )}</span>
    </div>`;

    const skillRows = (r.skill_matches || []).map((m) => ({
      item: m.skill,
      match_type: m.match_type,
      score: m.score ?? 0,
      explanation:
        m.resume_evidence || (m.match_type === 'gap' ? 'Missing from candidate profile' : ''),
    }));

    const dimBlocks = [];
    for (const d of r.dimension_scores || []) {
      const key = d.dimension || '';
      const isSkills = key === 'skill_match';
      let body = '';
      let wide = false;

      if (isSkills && skillRows.length) {
        body = matchTableHtml(skillRows, 'Skill');
        wide = true;
      } else if ((d.evidence || []).length) {
        body = evidenceListHtml(d.evidence);
        wide = true;
      }

      dimBlocks.push(`<div class="detail-section${wide ? ' detail-section--wide' : ''}" data-testid="dimension">
        <div class="detail-section-title">${esc(d.label || key)}
          <span class="ds-score ${scoreBand(d.raw_score)}">${Number(d.raw_score).toFixed(0)}/100</span>
        </div>
        <div class="detail-reasoning">${esc(d.reasoning || 'No reasoning provided.')}</div>
        ${body}
      </div>`);
    }

    html += `<div class="detail-grid">${dimBlocks.join('')}</div>`;

    if ((r.penalties || []).length) {
      html += `<div class="detail-section detail-section--wide">
        <div class="detail-section-title">Penalties</div>
        ${r.penalties
          .map(
            (p) => `<div class="penalty-card">
            <div style="flex:1;min-width:0">
              <div class="penalty-name">${esc(p.name)}</div>
              <div class="penalty-reason">${esc(p.reason)}</div>
            </div>
            <div class="penalty-scores">${Number(p.score_before).toFixed(1)} → ${Number(p.score_after).toFixed(1)}</div>
          </div>`,
          )
          .join('')}
      </div>`;
    }

    html += `<div class="match-legend">
      <div class="legend-grid">
        <div class="legend-item"><span class="match-type-badge mt-exact">EXACT</span><div class="legend-desc">Direct match with clear evidence.</div></div>
        <div class="legend-item"><span class="match-type-badge mt-transferable">TRANSFERABLE</span><div class="legend-desc">Related skill that transfers quickly.</div></div>
        <div class="legend-item"><span class="match-type-badge mt-partial">PARTIAL</span><div class="legend-desc">Some overlap, incomplete depth.</div></div>
        <div class="legend-item"><span class="match-type-badge mt-gap">GAP</span><div class="legend-desc">No meaningful evidence found.</div></div>
      </div>
    </div>`;

    if (r.trace && r.trace.spans && r.trace.spans.length) {
      html += `<details class="trace-fold">
        <summary>Execution trace — ${r.trace.spans.length} spans</summary>
        <div class="trace-list">
          ${r.trace.spans
            .map(
              (s) =>
                `<div data-testid="trace-span" class="trace-row"><span>${esc(s.name)} · ${esc(s.status)}</span><span>${esc(s.duration_ms)}ms</span></div>`,
            )
            .join('')}
        </div>
      </details>`;
    }

    return html;
  }

  function renderResult(result, isTop, idx) {
    const card = el(
      'div',
      `result-card overflow-hidden ${
        isTop ? 'border border-[#3ecf8e]/35 bg-[#111113]' : 'border border-white/8 bg-[#111113]'
      }`,
    );
    card.setAttribute('data-testid', 'result');
    if (isTop) card.setAttribute('data-top', 'true');

    const row = el('div', 'result-card-head flex items-center gap-3 sm:gap-4 px-4 py-3');

    const badge = el(
      'span',
      `inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-xs font-mono font-semibold ${
        isTop ? 'bg-[#3ecf8e] text-[#0a0a0b]' : 'bg-white/5 text-zinc-400'
      }`,
      `#${result.rank}`,
    );
    badge.setAttribute('data-testid', 'rank-badge');

    const dial = scoreDial(result.final_score);
    dial.setAttribute('data-testid', 'result-score');

    const idBox = el('div', 'flex-1 min-w-0');
    idBox.setAttribute('data-testid', 'result-id');
    idBox.appendChild(
      el(
        'div',
        'font-sans font-semibold text-zinc-100 truncate',
        result.candidate_name || result.source_file || 'Unnamed',
      ),
    );
    const subParts = [
      result.source_file,
      result.candidate_email,
      result.experience_level,
      result.candidate_years != null ? `${result.candidate_years} yrs` : null,
    ].filter(Boolean);
    if (subParts.length) {
      const sub = el('div', 'text-xs text-zinc-500 truncate mt-0.5', subParts.join(' · '));
      sub.setAttribute('data-testid', 'result-sub');
      idBox.appendChild(sub);
    }
    if (result.needs_human_review) {
      idBox.appendChild(
        el(
          'span',
          'mt-1 inline-flex rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-mono text-amber-300',
          'needs review',
        ),
      );
    }

    const matches = result.skill_matches || [];
    if (matches.length) {
      const peek = el('div', 'mt-1.5 flex flex-wrap gap-1');
      for (const m of matches.slice(0, 5)) {
        const chip = el(
          'span',
          `inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-mono ${CHIP[m.match_type] || 'border-white/10 text-zinc-300'}`,
        );
        chip.setAttribute('data-testid', 'skill-chip');
        chip.append(
          el('span', 'font-bold', GLYPH[m.match_type] || '?'),
          el('span', null, m.skill),
        );
        peek.appendChild(chip);
      }
      if (matches.length > 5) {
        peek.appendChild(el('span', 'text-[10px] text-zinc-600 self-center', `+${matches.length - 5}`));
      }
      idBox.appendChild(peek);
    }

    const panel = el('div', 'result-card-body');
    panel.setAttribute('data-testid', 'result-details');
    panel.hidden = false;
    panel.innerHTML = buildDetailHtml(result);
    card.classList.add('is-expanded');

    const detailsBtn = el('button', 'view-details-btn');
    detailsBtn.type = 'button';
    detailsBtn.textContent = 'Hide Details';
    detailsBtn.setAttribute('data-testid', 'view-details');
    detailsBtn.setAttribute('aria-expanded', 'true');
    detailsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const open = panel.hidden;
      panel.hidden = !open;
      detailsBtn.textContent = open ? 'Hide Details' : 'View Details';
      detailsBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
      card.classList.toggle('is-expanded', open);
    });

    row.append(badge, dial, idBox, detailsBtn);
    card.append(row, panel);
    return card;
  }

  function render(session) {
    resultsEl.replaceChildren();
    emptyEl.hidden = true;
    resultsEl.hidden = false;
    rankingResults = session.results || [];

    resultsEl.appendChild(renderSummary(session));

    if (session.job_profile) {
      const jp = session.job_profile;
      const musts = (jp.capabilities || []).filter((c) => c.importance === 'must_have');
      if (musts.length || jp.role_title) {
        const panel = el('div', 'rounded-lg border border-white/8 bg-[#111113] p-4');
        panel.appendChild(
          el('div', 'text-[10px] uppercase tracking-wider font-mono text-zinc-500 mb-3', 'Job profile'),
        );
        const dl = el('dl', 'grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm');
        if (jp.role_title) {
          dl.append(el('dt', 'font-mono text-[11px] text-zinc-500', 'role'), el('dd', null, jp.role_title));
        }
        if (jp.seniority) {
          dl.append(el('dt', 'font-mono text-[11px] text-zinc-500', 'seniority'), el('dd', null, jp.seniority));
        }
        if (jp.required_years_min != null) {
          dl.append(
            el('dt', 'font-mono text-[11px] text-zinc-500', 'min years'),
            el('dd', null, jp.required_years_min),
          );
        }
        panel.appendChild(dl);
        if (musts.length) {
          const list = el('div', 'flex flex-wrap gap-1.5 mt-3');
          for (const c of musts.slice(0, 24)) {
            list.appendChild(
              el(
                'span',
                'rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] font-mono text-zinc-300',
                c.name,
              ),
            );
          }
          panel.appendChild(list);
        }
        resultsEl.appendChild(panel);
      }
    }

    rankingResults.forEach((r, i) => {
      resultsEl.appendChild(renderResult(r, i === 0, i));
    });

    for (const err of session.processing_errors || []) {
      const n = el(
        'div',
        'flex gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 text-red-100 px-3 py-2.5 text-sm',
      );
      n.setAttribute('data-testid', 'processing-error');
      n.append(icon(ICONS.warn, 15), el('span', null, err));
      resultsEl.appendChild(n);
    }

    if (!rankingResults.length) {
      const e = el(
        'div',
        'rounded-2xl border border-dashed border-white/10 px-6 py-10 text-center text-sm text-zinc-400',
      );
      e.appendChild(el('p', null, 'No candidate could be scored. See the errors above.'));
      resultsEl.appendChild(e);
    }

    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function submit(event) {
    event.preventDefault();

    const jdText = jd.value.trim();

    if (jdText.length < 80) {
      setStatus('err', `The job description is too short (${jdText.length} chars, minimum 80).`);
      return;
    }
    if (!pendingFiles.length) {
      setStatus('err', 'Upload at least one resume (PDF or TXT).');
      return;
    }

    runBtn.disabled = true;
    const total = pendingFiles.length;
    setStatus('', `Running 7 agents across ${total} candidate${total === 1 ? '' : 's'}…`, true);

    const started = performance.now();
    try {
      const fd = new FormData();
      fd.append('jd_text', jdText);
      for (const f of pendingFiles) fd.append('files', f, f.name);

      const response = await fetch('/rank', { method: 'POST', body: fd });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detail =
          payload && payload.detail
            ? typeof payload.detail === 'string'
              ? payload.detail
              : JSON.stringify(payload.detail)
            : `HTTP ${response.status}`;
        setStatus('err', detail);
        return;
      }

      render(payload);
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      const scored = (payload.results || []).length;
      setStatus('ok', `Ranked ${scored} of ${payload.total_candidates} candidates in ${secs}s.`);
    } catch (err) {
      setStatus('err', `Request failed: ${err.message}. Is the server running?`);
    } finally {
      runBtn.disabled = false;
    }
  }

  wireDropzone();
  jd.addEventListener('input', updateCounts);
  $('rank-form').addEventListener('submit', submit);

  fetch('/healthz')
    .then((r) => r.json())
    .then((h) => {
      $('provider-badge').textContent = `${h.provider} · ${h.model}`;
    })
    .catch(() => {
      $('provider-badge').textContent = 'offline';
    });

  updateCounts();
})();
