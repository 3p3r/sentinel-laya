import { $ } from "zx";

export const REMOTE = process.env.DEEZX_HOST ?? "geekom@deezx";
export const REMOTE_PASS = process.env.DEEZX_PASSWORD ?? " ";
export const REMOTE_DIR = "sentinel-laya";
export const REMOTE_PY = `~/${REMOTE_DIR}/.venv/bin/python`;

const SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"];
const SSH_OPTS_STR = SSH_OPTS.join(" ");

/** Run a command on deezx, streaming output. */
export async function ssh(cmd: string) {
  return $`sshpass -p ${REMOTE_PASS} ssh ${SSH_OPTS} ${REMOTE} ${cmd}`;
}

/** Run a command on deezx quietly, returning trimmed stdout. */
export async function sshOut(cmd: string): Promise<string> {
  const p = await $`sshpass -p ${REMOTE_PASS} ssh ${SSH_OPTS} ${REMOTE} ${cmd}`.quiet();
  return p.stdout.trim();
}

/** rsync local -> remote (mirror code, never venvs/artifacts). */
export async function push(subdir = "") {
  const src = subdir ? `${subdir}/` : "./";
  const dst = `${REMOTE}:~/${REMOTE_DIR}/${subdir ? subdir + "/" : ""}`;
  await $`sshpass -p ${REMOTE_PASS} rsync -az --delete --exclude node_modules --exclude .venv --exclude .venv-stable --exclude .git --exclude data --exclude models --exclude results --exclude __pycache__ -e ${`ssh ${SSH_OPTS_STR}`} ${src} ${dst}`;
}

/** rsync remote -> local for produced artifacts (results, models, data reports). */
export async function pull(remoteSub: string, localSub: string) {
  await $`mkdir -p ${localSub}`;
  await $`sshpass -p ${REMOTE_PASS} rsync -az -e ${`ssh ${SSH_OPTS_STR}`} ${`${REMOTE}:~/${REMOTE_DIR}/${remoteSub}/`} ${`${localSub}/`}`;
}
