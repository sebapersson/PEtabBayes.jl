using Aqua, PEtabBayes

@testset "Aqua" begin
    Aqua.test_ambiguities(PEtabBayes, recursive = false)
    Aqua.test_undefined_exports(PEtabBayes)
    Aqua.test_unbound_args(PEtabBayes)
    Aqua.test_stale_deps(PEtabBayes)
    Aqua.test_deps_compat(PEtabBayes)
    Aqua.find_persistent_tasks_deps(PEtabBayes)
    Aqua.test_piracies(PEtabBayes)
    Aqua.test_project_extras(PEtabBayes)
    #Aqua.test_undocumented_names(PEtabBayes) enable when docs are up
end
