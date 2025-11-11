{{- define "online-service.fullname" -}}
{{ include "online-service.name" . }}
{{- end -}}

{{- define "online-service.name" -}}
{{ .Chart.Name }}
{{- end -}}

{{- define "online-service.labels" -}}
app.kubernetes.io/name: {{ include "online-service.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "online-service.selectorLabels" -}}
app.kubernetes.io/name: {{ include "online-service.name" . }}
{{- end -}}
