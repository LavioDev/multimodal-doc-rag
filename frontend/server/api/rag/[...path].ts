import { proxyRequest } from 'h3'

export default defineEventHandler((event) => {
  const config = useRuntimeConfig(event)
  const path = getRouterParam(event, 'path') || ''
  const target = `${String(config.ragApiBase).replace(/\/$/, '')}/${path}${getQuery(event) ? '' : ''}`
  const query = getRequestURL(event).search
  return proxyRequest(event, `${target}${query}`)
})
